﻿// Code for calculating NNUE evaluation function

#if defined(EVAL_NNUE)

#include <fstream>
#include <iostream>

#include "../../evaluate.h"
#include "../../position.h"
#include "../../misc.h"
#include "../../uci.h"

#include "evaluate_nnue.h"

namespace Eval {

	namespace NNUE {

		// Input feature converter
		AlignedPtr<FeatureTransformer> feature_transformer;

		// Evaluation function
		AlignedPtr<Network> network;

		// Evaluation function file name
		const char* const kFileName = "nn.bin";

		// Get a string that represents the structure of the evaluation function
		std::string GetArchitectureString() {
			return "Features=" + FeatureTransformer::GetStructureString() +
				",Network=" + Network::GetStructureString();
		}

		namespace {

			namespace Detail {

				// Initialize the evaluation function parameters
				template <typename T>
				void Initialize(AlignedPtr<T>& pointer) {
					pointer.reset(reinterpret_cast<T*>(aligned_malloc(sizeof(T), alignof(T))));
					std::memset(pointer.get(), 0, sizeof(T));
				}

				// read evaluation function parameters
				template <typename T>
				bool ReadParameters(std::istream& stream, const AlignedPtr<T>& pointer) {
					std::uint32_t header;
					stream.read(reinterpret_cast<char*>(&header), sizeof(header));
					if (!stream || header != T::GetHashValue()) return false;
					return pointer->ReadParameters(stream);
				}

				// write evaluation function parameters
				template <typename T>
				bool WriteParameters(std::ostream& stream, const AlignedPtr<T>& pointer) {
					constexpr std::uint32_t header = T::GetHashValue();
					stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
					return pointer->WriteParameters(stream);
				}

			} // namespace Detail

			// Initialize the evaluation function parameters
			void Initialize() {
				Detail::Initialize(feature_transformer);
				Detail::Initialize(network);
			}

		} // namespace

		// read the header
		bool ReadHeader(std::istream& stream,
			std::uint32_t* hash_value, std::string* architecture) {
			std::uint32_t version, size;
			stream.read(reinterpret_cast<char*>(&version), sizeof(version));
			stream.read(reinterpret_cast<char*>(hash_value), sizeof(*hash_value));
			stream.read(reinterpret_cast<char*>(&size), sizeof(size));
			if (!stream || version != kVersion) return false;
			architecture->resize(size);
			stream.read(&(*architecture)[0], size);
			return !stream.fail();
		}

		// write the header
		bool WriteHeader(std::ostream& stream,
			std::uint32_t hash_value, const std::string& architecture) {
			stream.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
			stream.write(reinterpret_cast<const char*>(&hash_value), sizeof(hash_value));
			const std::uint32_t size = static_cast<std::uint32_t>(architecture.size());
			stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
			stream.write(architecture.data(), size);
			return !stream.fail();
		}

		// read evaluation function parameters
		bool ReadParameters(std::istream& stream) {
			std::uint32_t hash_value;
			std::string architecture;
			if (!ReadHeader(stream, &hash_value, &architecture)) return false;
			if (hash_value != kHashValue) return false;
			if (!Detail::ReadParameters(stream, feature_transformer)) return false;
			if (!Detail::ReadParameters(stream, network)) return false;
			return stream && stream.peek() == std::ios::traits_type::eof();
		}

		// write evaluation function parameters
		bool WriteParameters(std::ostream& stream) {
			if (!WriteHeader(stream, kHashValue, GetArchitectureString())) return false;
			if (!Detail::WriteParameters(stream, feature_transformer)) return false;
			if (!Detail::WriteParameters(stream, network)) return false;
			return !stream.fail();
		}

		// proceed if you can calculate the difference
		static void UpdateAccumulatorIfPossible(const Position& pos) {
			feature_transformer->UpdateAccumulatorIfPossible(pos);
		}

		// calculate the evaluation value
		static Value ComputeScore(const Position& pos, bool refresh = false) {
			auto& accumulator = pos.state()->accumulator;
			if (!refresh && accumulator.computed_score) {
				return accumulator.score;
			}

			alignas(kCacheLineSize) TransformedFeatureType
				transformed_features[FeatureTransformer::kBufferSize];
			feature_transformer->Transform(pos, transformed_features, refresh);
			alignas(kCacheLineSize) char buffer[Network::kBufferSize];
			const auto output = network->Propagate(transformed_features, buffer);

			// aspiration search fails high when a value larger than VALUE_MAX_EVAL is returned
			// It should be guaranteed that it is less than or equal to VALUE_MAX_EVAL because the search will not end.

			// Even if this phenomenon occurs, if the seconds are fixed when playing, the search will be aborted there, so
			// The best move in the previous iteration is pointed to as bestmove, so apparently
			// no problem. The situation where this VALUE_MAX_EVAL is returned is almost at a dead end,
			// Since such a clogging phase often appears at the end, there is a big difference in the situation
			// Doesn't really affect the outcome.

			// However, when searching with a fixed depth such as when creating a teacher, it will not return from the search
			// waste computation time on that thread. In addition, it will be time-out at fixed depth game.

			auto score = static_cast<Value>(output[0] / FV_SCALE);

			// 1) I feel that if I clip too poorly, it will have an effect on my learning...
			// 2) Since accumulator.score is not used in the difference calculation, it can be rewritten without any problem.
			score = Math::clamp(score, -VALUE_MAX_EVAL, VALUE_MAX_EVAL);

			accumulator.score = score;
			accumulator.computed_score = true;
			return accumulator.score;
		}

	} // namespace NNUE

#if defined(USE_EVAL_HASH)
// Class used to store evaluation values ​​in HashTable
	struct alignas(16) ScoreKeyValue {
#if defined(USE_SSE2)
		ScoreKeyValue() = default;
		ScoreKeyValue(const ScoreKeyValue& other) {
			static_assert(sizeof(ScoreKeyValue) == sizeof(__m128i),
				"sizeof(ScoreKeyValue) should be equal to sizeof(__m128i)");
			_mm_store_si128(&as_m128i, other.as_m128i);
		}
		ScoreKeyValue& operator=(const ScoreKeyValue& other) {
			_mm_store_si128(&as_m128i, other.as_m128i);
			return *this;
		}
#endif

		// evaluate hashでatomicに操作できる必要があるのでそのための操作子
		void encode() {
#if defined(USE_SSE2)
			// ScoreKeyValue は atomic にコピーされるので key が合っていればデータも合っている。
#else
			key ^= score;
#endif
		}
		// decode()はencode()の逆変換だが、xorなので逆変換も同じ変換。
		void decode() { encode(); }

		union {
			struct {
				std::uint64_t key;
				std::uint64_t score;
			};
#if defined(USE_SSE2)
			__m128i as_m128i;
#endif
		};
	};

	// シンプルなHashTableの実装。
	// Sizeは2のべき乗。
	template <typename T, size_t Size>
	struct HashTable {
		HashTable() { clear(); }
		T* operator [] (const Key k) { return entries_ + (static_cast<size_t>(k) & (Size - 1)); }
		void clear() { memset(entries_, 0, sizeof(T) * Size); }

		// Check that Size is a power of 2
		static_assert((Size& (Size - 1)) == 0, "");

	private:
		T entries_[Size];
	};

	//HashTable to save the evaluated ones (the so-called ehash)

#if !defined(USE_LARGE_EVAL_HASH)
// 134MB (setting other than witch's AVX2)
	struct EvaluateHashTable : HashTable<ScoreKeyValue, 0x800000> {};
#else
// If you have prefetch, it's better to have a big one...
// → It doesn't change much, and it's a waste of memory, so the default setting is ↑.
// 1GB (setting for witch's AVX2)
	struct EvaluateHashTable : HashTable<ScoreKeyValue, 0x4000000> {};
#endif

	EvaluateHashTable g_evalTable;

	// Prepare a function to prefetch.
	void prefetch_evalhash(const Key key) {
		constexpr auto mask = ~((uint64_t)0x1f);
		prefetch((void*)((uint64_t)g_evalTable[key] & mask));
	}
#endif

	// Read the evaluation function file
	// Save and restore Options with bench command etc., so EvalDir has been changed at this time,
	// This function may be called twice to flag that the evaluation function needs to be reloaded.
	void load_eval() {
		NNUE::Initialize();

		if (!Options["SkipLoadingEval"])
		{
			const std::string dir_name = Options["EvalDir"];
			const std::string file_name = Path::Combine(dir_name, NNUE::kFileName);
			//{
			// std::ofstream stream(file_name, std::ios::binary);
			// NNUE::WriteParameters(stream);
			//}
			std::ifstream stream(file_name, std::ios::binary);
			const bool result = NNUE::ReadParameters(stream);

			// ASSERT(result);
			if (!result)
			{
				// I'm troubled if it doesn't finish when there is a read error.
				std::cout << "Error! : failed to read " << NNUE::kFileName << std::endl;
				my_exit();
			}
			else
				std::cout << "info string NNUE " << NNUE::kFileName << " found & loaded..." << std::endl;
		}
	}

	// Initialization
	void init() {
	}

	// Evaluation function. Perform full calculation instead of difference calculation.
	// Called only once with Position::set(). (The difference calculation after that)
	// Note that the evaluation value seen from the turn side is returned. (Design differs from other evaluation functions in this respect)
	// Since , we do not try to optimize this function.
	Value compute_eval(const Position& pos) {
		return NNUE::ComputeScore(pos, true);
	}

	// Evaluation function
	Value evaluate(const Position& pos) {
		const auto& accumulator = pos.state()->accumulator;
		if (accumulator.computed_score) {
			return accumulator.score;
		}

#if defined(USE_GLOBAL_OPTIONS)
		// If Global Options is set not to use eval hash
		// Skip the query to the eval hash.
		if (!GlobalOptions.use_eval_hash) {
			ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
			return NNUE::ComputeScore(pos);
		}
#endif

#if defined(USE_EVAL_HASH)
		// May be in the evaluate hash table.
		const Key key = pos.key();
		ScoreKeyValue entry = *g_evalTable[key];
		entry.decode();
		if (entry.key == key) {
			// there were!
			return Value(entry.score);
		}
#endif

		Value score = NNUE::ComputeScore(pos);
#if defined(USE_EVAL_HASH)
		// Since it was calculated carefully, save it in the evaluate hash table.
		entry.key = key;
		entry.score = score;
		entry.encode();
		*g_evalTable[key] = entry;
#endif

		return score;
	}

	// proceed if you can calculate the difference
	void evaluate_with_no_return(const Position& pos) {
		NNUE::UpdateAccumulatorIfPossible(pos);
	}

	// display the breakdown of the evaluation value of the current phase
	void print_eval_stat(Position& /*pos*/) {
		std::cout << "--- EVAL STAT: not implemented" << std::endl;
	}

} // namespace Eval

#endif // defined(EVAL_NNUE)