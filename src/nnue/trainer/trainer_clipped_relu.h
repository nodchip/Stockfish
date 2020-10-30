﻿#ifndef _NNUE_TRAINER_CLIPPED_RELU_H_
#define _NNUE_TRAINER_CLIPPED_RELU_H_

#include "trainer.h"

#include "learn/learn.h"

#include "nnue/layers/clipped_relu.h"

#include "thread.h"

// Specialization of NNUE evaluation function learning class template for ClippedReLU
namespace Eval::NNUE {

    // Learning: Affine transformation layer
    template <typename PreviousLayer>
    class Trainer<Layers::ClippedReLU<PreviousLayer>> {
    private:
        // Type of layer to learn
        using LayerType = Layers::ClippedReLU<PreviousLayer>;

    public:
        // factory function
        static std::shared_ptr<Trainer> create(
            LayerType* target_layer, FeatureTransformer* ft) {

            return std::shared_ptr<Trainer>(
                new Trainer(target_layer, ft));
        }

        // Set options such as hyperparameters
        void send_message(Message* message) {
            previous_layer_trainer_->send_message(message);
            if (receive_message("check_health", message)) {
                check_health();
            }
        }

        // Initialize the parameters with random numbers
        template <typename RNG>
        void initialize(RNG& rng) {
            previous_layer_trainer_->initialize(rng);
        }

        // forward propagation
        const LearnFloatType* propagate(ThreadPool& thread_pool, const std::vector<Example>& batch) {
            if (output_.size() < kOutputDimensions * batch.size()) {
              output_.resize(kOutputDimensions * batch.size());
              gradients_.resize(kInputDimensions * batch.size());
            }

            const auto input = previous_layer_trainer_->propagate(thread_pool, batch);

            batch_size_ = static_cast<IndexType>(batch.size());

#if defined (USE_SSE2)

            {
                static_assert(kOutputDimensions % 16 == 0, "This implementation assumes that it can process 16 floats at a time");

                const __m128 kZero4 = _mm_set1_ps(+kZero);
                const __m128 kOne4 = _mm_set1_ps(+kOne);

                for (IndexType b = 0; b < batch.size(); ++b)
                {
                    const IndexType batch_offset = kOutputDimensions * b;

                    for (IndexType i = 0; i < kOutputDimensions; i += 16)
                    {
                        __m128 out0 = _mm_loadu_ps(&input[i + 0 + batch_offset]);
                        __m128 out1 = _mm_loadu_ps(&input[i + 4 + batch_offset]);
                        __m128 out2 = _mm_loadu_ps(&input[i + 8 + batch_offset]);
                        __m128 out3 = _mm_loadu_ps(&input[i + 12 + batch_offset]);

                        out0 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out0));
                        out1 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out1));
                        out2 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out2));
                        out3 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out3));

                        _mm_storeu_ps(&output_[i + 0 + batch_offset], out0);
                        _mm_storeu_ps(&output_[i + 4 + batch_offset], out1);
                        _mm_storeu_ps(&output_[i + 8 + batch_offset], out2);
                        _mm_storeu_ps(&output_[i + 12 + batch_offset], out3);

                        __m128 minact0 = _mm_loadu_ps(&min_activations_[i + 0]);
                        __m128 minact1 = _mm_loadu_ps(&min_activations_[i + 4]);
                        __m128 minact2 = _mm_loadu_ps(&min_activations_[i + 8]);
                        __m128 minact3 = _mm_loadu_ps(&min_activations_[i + 12]);

                        __m128 maxact0 = _mm_loadu_ps(&max_activations_[i + 0]);
                        __m128 maxact1 = _mm_loadu_ps(&max_activations_[i + 4]);
                        __m128 maxact2 = _mm_loadu_ps(&max_activations_[i + 8]);
                        __m128 maxact3 = _mm_loadu_ps(&max_activations_[i + 12]);

                        minact0 = _mm_min_ps(out0, minact0);
                        minact1 = _mm_min_ps(out1, minact1);
                        minact2 = _mm_min_ps(out2, minact2);
                        minact3 = _mm_min_ps(out3, minact3);

                        maxact0 = _mm_max_ps(out0, maxact0);
                        maxact1 = _mm_max_ps(out1, maxact1);
                        maxact2 = _mm_max_ps(out2, maxact2);
                        maxact3 = _mm_max_ps(out3, maxact3);

                        _mm_storeu_ps(&min_activations_[i + 0], minact0);
                        _mm_storeu_ps(&min_activations_[i + 4], minact1);
                        _mm_storeu_ps(&min_activations_[i + 8], minact2);
                        _mm_storeu_ps(&min_activations_[i + 12], minact3);

                        _mm_storeu_ps(&max_activations_[i + 0], maxact0);
                        _mm_storeu_ps(&max_activations_[i + 4], maxact1);
                        _mm_storeu_ps(&max_activations_[i + 8], maxact2);
                        _mm_storeu_ps(&max_activations_[i + 12], maxact3);
                    }
                }
            }

#else

            for (IndexType b = 0; b < batch_size_; ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                for (IndexType i = 0; i < kOutputDimensions; ++i) {
                    const IndexType index = batch_offset + i;
                    output_[index] = std::max(+kZero, std::min(+kOne, input[index]));
                    min_activations_[i] = std::min(min_activations_[i], output_[index]);
                    max_activations_[i] = std::max(max_activations_[i], output_[index]);
                }
            }

#endif

            return output_.data();
        }

        // backpropagation
        void backpropagate(ThreadPool& thread_pool,
                           const LearnFloatType* gradients,
                           LearnFloatType learning_rate) {

            for (IndexType b = 0; b < batch_size_; ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                for (IndexType i = 0; i < kOutputDimensions; ++i) {
                    const IndexType index = batch_offset + i;
                    const bool clipped = (output_[index] <= kZero) | (output_[index] >= kOne);
                    gradients_[index] = gradients[index] * !clipped;
                    num_clipped_ += clipped;
                }
            }
            num_total_ += batch_size_ * kOutputDimensions;

            previous_layer_trainer_->backpropagate(thread_pool, gradients_.data(), learning_rate);
        }

    private:
        // constructor
        Trainer(LayerType* target_layer, FeatureTransformer* ft) :
            batch_size_(0),
            previous_layer_trainer_(Trainer<PreviousLayer>::create(
                &target_layer->previous_layer_, ft)),
            target_layer_(target_layer) {

            reset_stats();
        }

        void reset_stats() {
            std::fill(std::begin(min_activations_), std::end(min_activations_),
                      std::numeric_limits<LearnFloatType>::max());
            std::fill(std::begin(max_activations_), std::end(max_activations_),
                      std::numeric_limits<LearnFloatType>::lowest());

            num_clipped_ = 0;
            num_total_ = 0;
        }

        // Check if there are any problems with learning
        void check_health() {

            const auto largest_min_activation = *std::max_element(
                std::begin(min_activations_), std::end(min_activations_));
            const auto smallest_max_activation = *std::min_element(
                std::begin(max_activations_), std::end(max_activations_));

            auto out = sync_region_cout.new_region();

            out << "INFO (check_health):"
                << " layer " << LayerType::kLayerIndex
                << " - " << LayerType::get_name()
                << std::endl;

            out << "  - largest min activation = " << largest_min_activation
                << " , smallest max activation = " << smallest_max_activation
                << std::endl;

            out << "  - clipped " << static_cast<double>(num_clipped_) / num_total_ * 100.0 << "% of outputs"
                << std::endl;

            out.unlock();

            reset_stats();
        }

        // number of input/output dimensions
        static constexpr IndexType kInputDimensions = LayerType::kOutputDimensions;
        static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;

        // LearnFloatType constant
        static constexpr LearnFloatType kZero = static_cast<LearnFloatType>(0.0);
        static constexpr LearnFloatType kOne = static_cast<LearnFloatType>(1.0);

        // number of samples in mini-batch
        IndexType batch_size_;

        IndexType num_clipped_;
        IndexType num_total_;

        // Trainer of the previous layer
        const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

        // layer to learn
        LayerType* const target_layer_;

        // Forward propagation buffer
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> output_;

        // buffer for back propagation
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> gradients_;

        // Health check statistics
        LearnFloatType min_activations_[kOutputDimensions];
        LearnFloatType max_activations_[kOutputDimensions];
    };

}  // namespace Eval::NNUE

#endif
