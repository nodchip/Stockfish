﻿#ifndef _NNUE_TRAINER_FEATURE_TRANSFORMER_H_
#define _NNUE_TRAINER_FEATURE_TRANSFORMER_H_

#include "trainer.h"

#include "extra/stockfish_blas.h"

#include "features/factorizer_feature_set.h"

#include "learn/learn.h"

#include "nnue/nnue_feature_transformer.h"

#include "thread.h"

#include <array>
#include <bitset>
#include <numeric>
#include <random>
#include <set>

// Specialization for feature transformer of learning class template of NNUE evaluation function
namespace Eval::NNUE {

    // Learning: Input feature converter
    template <>
    class Trainer<FeatureTransformer> {
    private:
        // Type of layer to learn
        using LayerType = FeatureTransformer;

    public:
        template <typename T>
        friend struct AlignedDeleter;

        template <typename T, typename... ArgumentTypes>
        friend std::shared_ptr<T> make_aligned_shared_ptr(ArgumentTypes&&... arguments);

        // factory function
        static std::shared_ptr<Trainer> create(LayerType* target_layer) {
            return make_aligned_shared_ptr<Trainer>(target_layer);
        }

        // Set options such as hyperparameters
        void send_message(Message* message) {
            if (receive_message("momentum", message)) {
                momentum_ = static_cast<LearnFloatType>(std::stod(message->value));
            }

            if (receive_message("learning_rate_scale", message)) {
                learning_rate_scale_ =
                    static_cast<LearnFloatType>(std::stod(message->value));
            }

            if (receive_message("reset", message)) {
                dequantize_parameters();
            }

            if (receive_message("quantize_parameters", message)) {
                quantize_parameters();
            }

            if (receive_message("clear_unobserved_feature_weights", message)) {
                clear_unobserved_feature_weights();
            }

            if (receive_message("check_health", message)) {
                check_health();
            }
        }

        // Initialize the parameters with random numbers
        template <typename RNG>
        void initialize(RNG& rng) {
            std::fill(std::begin(weights_), std::end(weights_), +kZero);

            const double kSigma = 0.1 / std::sqrt(RawFeatures::kMaxActiveDimensions);
            auto distribution = std::normal_distribution<double>(0.0, kSigma);

            for (IndexType i = 0; i < kHalfDimensions * RawFeatures::kDimensions; ++i) {
                const auto weight = static_cast<LearnFloatType>(distribution(rng));
                weights_[i] = weight;
            }

            for (IndexType i = 0; i < kHalfDimensions; ++i) {
                biases_[i] = static_cast<LearnFloatType>(0.5);
            }

            quantize_parameters();
        }

        // forward propagation
        const LearnFloatType* propagate(ThreadPool& thread_pool, const std::vector<Example>& batch) {
            if (output_.size() < kOutputDimensions * batch.size()) {
                output_.resize(kOutputDimensions * batch.size());
                gradients_.resize(kOutputDimensions * batch.size());
            }

            (void)thread_pool;

            batch_ = &batch;
            // affine transform
            thread_pool.for_each_index_with_workers(
                0, batch.size(),
                [&](Thread&, int b) {
                    const IndexType batch_offset = kOutputDimensions * b;
                    for (IndexType c = 0; c < 2; ++c) {
                        const IndexType output_offset = batch_offset + kHalfDimensions * c;

#if defined(USE_BLAS)

                        cblas_scopy(
                            kHalfDimensions, biases_, 1, &output_[output_offset], 1
                        );

                        for (const auto& feature : batch[b].training_features[c]) {
                            const IndexType weights_offset = kHalfDimensions * feature.get_index();
                            cblas_saxpy(
                                kHalfDimensions, (float)feature.get_count(),
                                &weights_[weights_offset], 1, &output_[output_offset], 1
                            );
                        }

#else

                        Blas::scopy(
                            kHalfDimensions, biases_, 1, &output_[output_offset], 1
                        );
                        for (const auto& feature : batch[b].training_features[c]) {
                            const IndexType weights_offset = kHalfDimensions * feature.get_index();
                            Blas::saxpy(
                                kHalfDimensions, (float)feature.get_count(),
                                &weights_[weights_offset], 1, &output_[output_offset], 1
                            );
                        }

#endif
                    }
                }
            );
            thread_pool.wait_for_workers_finished();

            // clipped ReLU
            for (IndexType b = 0; b < batch.size(); ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                for (IndexType i = 0; i < kOutputDimensions; ++i) {
                    const IndexType index = batch_offset + i;
                    min_pre_activation_ = std::min(min_pre_activation_, output_[index]);
                    max_pre_activation_ = std::max(max_pre_activation_, output_[index]);
                    output_[index] = std::max(+kZero, std::min(+kOne, output_[index]));
                    const IndexType t = i % kHalfDimensions;
                    min_activations_[t] = std::min(min_activations_[t], output_[index]);
                    max_activations_[t] = std::max(max_activations_[t], output_[index]);
                }
            }

            return output_.data();
        }

        // backpropagation
        void backpropagate(ThreadPool& thread_pool,
                           const LearnFloatType* gradients,
                           LearnFloatType learning_rate) {

            (void)thread_pool;

            const LearnFloatType local_learning_rate =
                learning_rate * learning_rate_scale_;

            for (IndexType b = 0; b < batch_->size(); ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                for (IndexType i = 0; i < kOutputDimensions; ++i) {
                    const IndexType index = batch_offset + i;
                    const bool clipped = (output_[index] <= kZero) | (output_[index] >= kOne);
                    gradients_[index] = gradients[index] * !clipped;
                    num_clipped_ += clipped;
                }
            }
            num_total_ += batch_->size() * kOutputDimensions;

            // Since the weight matrix updates only the columns corresponding to the features that appeared in the input,
            // Correct the learning rate and adjust the scale without using momentum
            const LearnFloatType effective_learning_rate =
                static_cast<LearnFloatType>(local_learning_rate / (1.0 - momentum_));

#if defined(USE_BLAS)

            cblas_sscal(
                kHalfDimensions, momentum_, biases_diff_, 1
            );

            for (IndexType b = 0; b < batch_->size(); ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                for (IndexType c = 0; c < 2; ++c) {
                    const IndexType output_offset = batch_offset + kHalfDimensions * c;
                    cblas_saxpy(
                        kHalfDimensions, 1.0,
                        &gradients_[output_offset], 1, biases_diff_, 1
                    );
                }
            }

            cblas_saxpy(
                kHalfDimensions, -local_learning_rate,
                biases_diff_, 1, biases_, 1
            );

#else

            Blas::sscal(
                thread_pool,
                kHalfDimensions, momentum_, biases_diff_, 1
            );

            for (IndexType b = 0; b < batch_->size(); ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                for (IndexType c = 0; c < 2; ++c) {
                    const IndexType output_offset = batch_offset + kHalfDimensions * c;
                    Blas::saxpy(
                        thread_pool,
                        kHalfDimensions, 1.0,
                        &gradients_[output_offset], 1, biases_diff_, 1
                    );
                }
            }

            Blas::saxpy(
                thread_pool,
                kHalfDimensions, -local_learning_rate,
                biases_diff_, 1, biases_, 1
            );

#endif

            thread_pool.execute_with_workers(
                [&, num_threads = thread_pool.size()](Thread& th) {
                    const auto thread_index = th.thread_idx();

                    for (IndexType b = 0; b < batch_->size(); ++b) {
                        const IndexType batch_offset = kOutputDimensions * b;
                        for (IndexType c = 0; c < 2; ++c) {
                            const IndexType output_offset = batch_offset + kHalfDimensions * c;
                            for (const auto& feature : (*batch_)[b].training_features[c]) {
                                if (feature.get_index() % num_threads != thread_index)
                                    continue;
                                const IndexType weights_offset =
                                    kHalfDimensions * feature.get_index();
                                const auto scale = static_cast<LearnFloatType>(
                                    effective_learning_rate / feature.get_count());

#if defined (USE_BLAS)

                                cblas_saxpy(
                                    kHalfDimensions, -scale,
                                    &gradients_[output_offset], 1,
                                    &weights_[weights_offset], 1
                                );

#else

                                Blas::saxpy(
                                    kHalfDimensions, -scale,
                                    &gradients_[output_offset], 1,
                                    &weights_[weights_offset], 1
                                );

#endif
                            }
                        }
                    }
                }
            );

            for (IndexType b = 0; b < batch_->size(); ++b) {
                for (IndexType c = 0; c < 2; ++c) {
                    for (const auto& feature : (*batch_)[b].training_features[c]) {
                        observed_features.set(feature.get_index());
                    }
                }
            }

            thread_pool.wait_for_workers_finished();
        }

    private:
        // constructor
        Trainer(LayerType* target_layer) :
            batch_(nullptr),
            target_layer_(target_layer),
            biases_(),
            weights_(),
            biases_diff_(),
            momentum_(0.2),
            learning_rate_scale_(1.0) {

            dequantize_parameters();
        }

        // Weight saturation and parameterization
        void quantize_parameters() {
            for (IndexType i = 0; i < kHalfDimensions; ++i) {
                target_layer_->biases_[i] =
                    round<typename LayerType::BiasType>(biases_[i] * kBiasScale);
            }

            std::vector<TrainingFeature> training_features;

            Threads.for_each_index_with_workers(
                0, RawFeatures::kDimensions,
                [this, training_features](Thread&, int j) mutable {
                    training_features.clear();
                    Features::Factorizer<RawFeatures>::append_training_features(
                        j, &training_features);

                    for (IndexType i = 0; i < kHalfDimensions; ++i) {
                        double sum = 0.0;
                        for (const auto& feature : training_features) {
                            sum += weights_[kHalfDimensions * feature.get_index() + i];
                        }

                        target_layer_->weights_[kHalfDimensions * j + i] =
                            round<typename LayerType::WeightType>(sum * kWeightScale);
                    }
                }
            );
            Threads.wait_for_workers_finished();
        }

        void reset_stats() {
            min_pre_activation_ = std::numeric_limits<LearnFloatType>::max();
            max_pre_activation_ = std::numeric_limits<LearnFloatType>::lowest();

            std::fill(std::begin(min_activations_), std::end(min_activations_),
                      std::numeric_limits<LearnFloatType>::max());
            std::fill(std::begin(max_activations_), std::end(max_activations_),
                      std::numeric_limits<LearnFloatType>::lowest());

            num_clipped_ = 0;
            num_total_ = 0;
        }

        // read parameterized integer
        void dequantize_parameters() {
            for (IndexType i = 0; i < kHalfDimensions; ++i) {
                biases_[i] = static_cast<LearnFloatType>(
                    target_layer_->biases_[i] / kBiasScale);
            }

            std::fill(std::begin(weights_), std::end(weights_), +kZero);

            for (IndexType i = 0; i < kHalfDimensions * RawFeatures::kDimensions; ++i) {
                weights_[i] = static_cast<LearnFloatType>(
                    target_layer_->weights_[i] / kWeightScale);
            }

            std::fill(std::begin(biases_diff_), std::end(biases_diff_), +kZero);

            reset_stats();
        }

        // Set the weight corresponding to the feature that does not appear in the learning data to 0
        void clear_unobserved_feature_weights() {
            for (IndexType i = 0; i < kInputDimensions; ++i) {
                if (!observed_features.test(i)) {
                    std::fill(std::begin(weights_) + kHalfDimensions * i,
                              std::begin(weights_) + kHalfDimensions * (i + 1), +kZero);
                }
            }

            quantize_parameters();
        }

        // Check if there are any problems with learning
        void check_health() {

            constexpr LearnFloatType kPreActivationLimit =
                std::numeric_limits<typename LayerType::WeightType>::max() /
                kWeightScale;

            const auto largest_min_activation = *std::max_element(
                std::begin(min_activations_), std::end(min_activations_));
            const auto smallest_max_activation = *std::min_element(
                std::begin(max_activations_), std::end(max_activations_));

            double abs_bias_sum = 0.0;
            double abs_weight_sum = 0.0;

            for(auto b : biases_)
                abs_bias_sum += std::abs(b);

            for(auto w : weights_)
                abs_weight_sum += std::abs(w);

            auto out = sync_region_cout.new_region();

            out << "INFO (check_health):"
                << " layer " << LayerType::kLayerIndex
                << " - " << LayerType::get_name()
                << std::endl;

            out << "  - observed " << observed_features.count()
                << " (out of " << kInputDimensions << ") features"
                << std::endl;

            out << "  - (min, max) of pre-activations = "
                << min_pre_activation_ << ", "
                << max_pre_activation_ << " (limit = "
                << kPreActivationLimit << ")"
                << std::endl;

            out << "  - largest min activation = " << largest_min_activation
                << " , smallest max activation = " << smallest_max_activation
                << std::endl;

            out << "  - avg_abs_bias   = " << abs_bias_sum / std::size(biases_) << std::endl;
            out << "  - avg_abs_weight = " << abs_weight_sum / std::size(weights_) << std::endl;

            out << "  - clipped " << static_cast<double>(num_clipped_) / num_total_ * 100.0 << "% of outputs"
                << std::endl;

            out.unlock();

            reset_stats();
        }

        // number of input/output dimensions
        static constexpr IndexType kInputDimensions =
            Features::Factorizer<RawFeatures>::get_dimensions();
        static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;
        static constexpr IndexType kHalfDimensions = LayerType::kHalfDimensions;

        // Coefficient used for parameterization
        static constexpr LearnFloatType kActivationScale =
            std::numeric_limits<std::int8_t>::max();
        static constexpr LearnFloatType kBiasScale = kActivationScale;
        static constexpr LearnFloatType kWeightScale = kActivationScale;

        // LearnFloatType constant
        static constexpr LearnFloatType kZero = static_cast<LearnFloatType>(0.0);
        static constexpr LearnFloatType kOne = static_cast<LearnFloatType>(1.0);

        // mini batch
        const std::vector<Example>* batch_;

        // layer to learn
        LayerType* const target_layer_;

        IndexType num_clipped_;
        IndexType num_total_;

        // parameter
        alignas(kCacheLineSize) LearnFloatType biases_[kHalfDimensions];
        alignas(kCacheLineSize)
            LearnFloatType weights_[kHalfDimensions * kInputDimensions];

        // Buffer used for updating parameters
        alignas(kCacheLineSize) LearnFloatType biases_diff_[kHalfDimensions];
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> gradients_;

        // Forward propagation buffer
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> output_;

        // Features that appeared in the training data
        std::bitset<kInputDimensions> observed_features;

        // hyper parameter
        LearnFloatType momentum_;
        LearnFloatType learning_rate_scale_;

        // Health check statistics
        LearnFloatType min_pre_activation_;
        LearnFloatType max_pre_activation_;
        alignas(kCacheLineSize) LearnFloatType min_activations_[kHalfDimensions];
        alignas(kCacheLineSize) LearnFloatType max_activations_[kHalfDimensions];
    };

}  // namespace Eval::NNUE

#endif
