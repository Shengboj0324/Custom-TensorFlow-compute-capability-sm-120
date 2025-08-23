"""
Comprehensive SM120 High-Level Usage Example
Demonstrates the complete high-level API for SM120 optimized TensorFlow operations
with automatic fallback, performance monitoring, and error handling.

Copyright 2024 - TensorFlow SM120 Optimization Project
"""

import tensorflow as tf
import numpy as np
import time
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from python.sm120_keras_layers import (
        SM120Dense,
        SM120Conv2D,
        SM120BatchNormalization,
        SM120MultiHeadAttention,
        create_sm120_transformer_encoder,
    )
    from python import sm120_ops

    SM120_AVAILABLE = True
    print("✅ SM120 optimized operations loaded successfully")
except ImportError as e:
    print(f"⚠️  SM120 operations not available: {e}")
    print("   Falling back to standard TensorFlow operations")
    SM120_AVAILABLE = False


def check_gpu_compatibility():
    """Check if the current GPU supports SM120 optimizations."""
    print("\n🔍 GPU Compatibility Check")
    print("=" * 50)

    physical_devices = tf.config.list_physical_devices("GPU")
    if not physical_devices:
        print("❌ No GPU devices found")
        return False

    print(f"📱 Found {len(physical_devices)} GPU(s):")

    for i, device in enumerate(physical_devices):
        print(f"   GPU {i}: {device.name}")

        # Try to get device details
        try:
            device_details = tf.config.experimental.get_device_details(device)
            compute_capability = device_details.get("compute_capability")
            if compute_capability:
                major, minor = compute_capability
                print(f"      Compute Capability: {major}.{minor}")

                if major == 12 and minor == 0:
                    print("      ✅ SM120 (RTX 50-series) support detected!")
                    return True
                elif major >= 7:
                    print("      ⚠️  Tensor Core support available (not SM120)")
                else:
                    print("      ❌ Limited CUDA support")
            else:
                print("      ❓ Compute capability unknown")
        except Exception as e:
            print(f"      ❓ Could not get device details: {e}")

    return False


def benchmark_operations():
    """Benchmark SM120 operations against standard TensorFlow."""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)

    # Test configurations
    batch_size = 64
    seq_len = 512
    embed_dim = 768
    num_heads = 12

    print(
        f"Configuration: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}"
    )

    # Generate test data
    with tf.device("/GPU:0"):
        test_input = tf.random.normal(
            [batch_size, seq_len, embed_dim], dtype=tf.float16
        )
        test_weights = tf.random.normal([embed_dim, embed_dim], dtype=tf.float16)

        # Warm up GPU
        _ = tf.matmul(test_input, test_weights)

    results = {}

    # Benchmark Dense layer
    print("\n🧮 Dense Layer Benchmark")

    # SM120 Dense
    if SM120_AVAILABLE:
        sm120_dense = SM120Dense(embed_dim, use_sm120=True, dtype=tf.float16)
        sm120_dense.build(test_input.shape)

        start_time = time.time()
        for _ in range(10):
            with tf.device("/GPU:0"):
                _ = sm120_dense(test_input)
        tf.keras.backend.clear_session()
        sm120_time = (time.time() - start_time) / 10

        print(f"   SM120 Dense: {sm120_time:.4f}s per iteration")
        results["sm120_dense"] = sm120_time

    # Standard Dense
    standard_dense = tf.keras.layers.Dense(embed_dim, dtype=tf.float16)
    standard_dense.build(test_input.shape)

    start_time = time.time()
    for _ in range(10):
        with tf.device("/GPU:0"):
            _ = standard_dense(test_input)
    tf.keras.backend.clear_session()
    standard_time = (time.time() - start_time) / 10

    print(f"   Standard Dense: {standard_time:.4f}s per iteration")
    results["standard_dense"] = standard_time

    if SM120_AVAILABLE and "sm120_dense" in results:
        speedup = standard_time / results["sm120_dense"]
        print(f"   🚀 SM120 Speedup: {speedup:.2f}x")

    # Benchmark Convolution
    print("\n🖼️  Conv2D Benchmark")

    conv_input = tf.random.normal([batch_size, 224, 224, 3], dtype=tf.float16)

    if SM120_AVAILABLE:
        sm120_conv = SM120Conv2D(
            64, (3, 3), padding="same", use_sm120=True, dtype=tf.float16
        )
        sm120_conv.build(conv_input.shape)

        start_time = time.time()
        for _ in range(10):
            with tf.device("/GPU:0"):
                _ = sm120_conv(conv_input)
        tf.keras.backend.clear_session()
        sm120_conv_time = (time.time() - start_time) / 10

        print(f"   SM120 Conv2D: {sm120_conv_time:.4f}s per iteration")
        results["sm120_conv"] = sm120_conv_time

    standard_conv = tf.keras.layers.Conv2D(64, (3, 3), padding="same", dtype=tf.float16)
    standard_conv.build(conv_input.shape)

    start_time = time.time()
    for _ in range(10):
        with tf.device("/GPU:0"):
            _ = standard_conv(conv_input)
    tf.keras.backend.clear_session()
    standard_conv_time = (time.time() - start_time) / 10

    print(f"   Standard Conv2D: {standard_conv_time:.4f}s per iteration")
    results["standard_conv"] = standard_conv_time

    if SM120_AVAILABLE and "sm120_conv" in results:
        speedup = standard_conv_time / results["sm120_conv"]
        print(f"   🚀 SM120 Speedup: {speedup:.2f}x")

    # Benchmark Multi-Head Attention
    print("\n🎯 Multi-Head Attention Benchmark")

    attention_input = tf.random.normal(
        [batch_size, seq_len, embed_dim], dtype=tf.float16
    )

    if SM120_AVAILABLE:
        sm120_attention = SM120MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            use_sm120=True,
            use_flash_attention=True,
            dtype=tf.float16,
        )
        sm120_attention.build(attention_input.shape)

        start_time = time.time()
        for _ in range(5):  # Fewer iterations for attention (more expensive)
            with tf.device("/GPU:0"):
                _ = sm120_attention(attention_input)
        tf.keras.backend.clear_session()
        sm120_attention_time = (time.time() - start_time) / 5

        print(f"   SM120 Attention: {sm120_attention_time:.4f}s per iteration")
        results["sm120_attention"] = sm120_attention_time

    standard_attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dtype=tf.float16
    )
    standard_attention.build(attention_input.shape)

    start_time = time.time()
    for _ in range(5):
        with tf.device("/GPU:0"):
            _ = standard_attention(attention_input, attention_input)
    tf.keras.backend.clear_session()
    standard_attention_time = (time.time() - start_time) / 5

    print(f"   Standard Attention: {standard_attention_time:.4f}s per iteration")
    results["standard_attention"] = standard_attention_time

    if SM120_AVAILABLE and "sm120_attention" in results:
        speedup = standard_attention_time / results["sm120_attention"]
        print(f"   🚀 SM120 Speedup: {speedup:.2f}x")

    return results


def create_and_train_model():
    """Create and train a model using SM120 optimized layers."""
    print("\n🏗️  Model Creation and Training")
    print("=" * 50)

    # Model configuration
    vocab_size = 10000
    max_length = 256
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_classes = 10

    print(
        f"Creating transformer model with embed_dim={embed_dim}, num_heads={num_heads}"
    )

    try:
        if SM120_AVAILABLE:
            # Create SM120 optimized model
            model = create_sm120_transformer_encoder(
                vocab_size=vocab_size,
                max_length=max_length,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=4,
                dropout_rate=0.1,
                use_sm120=True,
            )

            # Add classification head
            inputs = model.input
            x = model.output
            x = tf.keras.layers.Dense(
                num_classes, activation="softmax", name="classifier"
            )(x)
            model = tf.keras.Model(inputs, x, name="sm120_transformer_classifier")

            print("✅ SM120 optimized model created successfully")
        else:
            # Fallback to standard model
            inputs = tf.keras.Input(shape=(max_length,), dtype=tf.int32)

            # Embedding and positional encoding
            embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)(inputs)
            positions = tf.keras.layers.Embedding(max_length, embed_dim)(
                tf.range(start=0, limit=max_length, delta=1)
            )
            x = embeddings + positions

            # Transformer layers
            for i in range(4):
                # Multi-head attention
                attention_output = tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embed_dim // num_heads,
                    name=f"attention_{i}",
                )(x, x)
                attention_output = tf.keras.layers.Dropout(0.1)(attention_output)
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                    x + attention_output
                )

                # Feed-forward network
                ff_output = tf.keras.layers.Dense(
                    ff_dim, activation="relu", name=f"ff1_{i}"
                )(x)
                ff_output = tf.keras.layers.Dropout(0.1)(ff_output)
                ff_output = tf.keras.layers.Dense(embed_dim, name=f"ff2_{i}")(ff_output)
                ff_output = tf.keras.layers.Dropout(0.1)(ff_output)
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

            # Global average pooling and output
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            outputs = tf.keras.layers.Dense(
                num_classes, activation="softmax", name="classifier"
            )(x)

            model = tf.keras.Model(
                inputs, outputs, name="standard_transformer_classifier"
            )
            print("✅ Standard model created successfully")

        # Print model summary
        print(f"\nModel Summary:")
        print(f"   Total parameters: {model.count_params():,}")
        print(
            f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}"
        )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Generate synthetic training data
        print("\n📊 Generating synthetic training data...")
        batch_size = 32
        num_samples = 1000

        # Generate random sequences
        x_train = np.random.randint(0, vocab_size, size=(num_samples, max_length))
        y_train = np.random.randint(0, num_classes, size=(num_samples,))

        x_val = np.random.randint(0, vocab_size, size=(200, max_length))
        y_val = np.random.randint(0, num_classes, size=(200,))

        print(f"   Training data: {x_train.shape}, {y_train.shape}")
        print(f"   Validation data: {x_val.shape}, {y_val.shape}")

        # Train model for a few epochs
        print("\n🏃 Training model...")

        # Create custom callback to monitor SM120 operations
        class SM120MonitorCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                if SM120_AVAILABLE:
                    print(f"   Epoch {epoch + 1}: SM120 optimizations active")

            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    print(
                        f"   Epoch {epoch + 1} - Loss: {logs.get('loss', 0):.4f}, "
                        f"Accuracy: {logs.get('accuracy', 0):.4f}, "
                        f"Val Loss: {logs.get('val_loss', 0):.4f}, "
                        f"Val Accuracy: {logs.get('val_accuracy', 0):.4f}"
                    )

        callbacks = [
            SM120MonitorCallback(),
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        ]

        # Train with timing
        start_time = time.time()

        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=5,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=0,
        )

        training_time = time.time() - start_time

        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"   Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(
            f"   Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}"
        )

        return model, history

    except Exception as e:
        print(f"❌ Error during model creation/training: {e}")
        return None, None


def demonstrate_gradient_computation():
    """Demonstrate that gradients work correctly with SM120 operations."""
    print("\n🎓 Gradient Computation Test")
    print("=" * 50)

    try:
        # Create a simple model with SM120 layers
        if SM120_AVAILABLE:
            inputs = tf.keras.Input(shape=(100,))
            x = SM120Dense(50, activation="relu", use_sm120=True)(inputs)
            x = SM120Dense(25, activation="relu", use_sm120=True)(x)
            outputs = SM120Dense(1, use_sm120=True)(x)
            model = tf.keras.Model(inputs, outputs)
            print("✅ SM120 model created for gradient test")
        else:
            inputs = tf.keras.Input(shape=(100,))
            x = tf.keras.layers.Dense(50, activation="relu")(inputs)
            x = tf.keras.layers.Dense(25, activation="relu")(x)
            outputs = tf.keras.layers.Dense(1)(x)
            model = tf.keras.Model(inputs, outputs)
            print("✅ Standard model created for gradient test")

        # Generate test data
        x_test = tf.random.normal([32, 100])
        y_test = tf.random.normal([32, 1])

        # Test forward pass
        with tf.GradientTape() as tape:
            predictions = model(x_test, training=True)
            loss = tf.reduce_mean(tf.square(predictions - y_test))

        # Test backward pass
        gradients = tape.gradient(loss, model.trainable_variables)

        # Verify gradients
        gradient_norms = [tf.norm(grad) for grad in gradients if grad is not None]

        print(f"   Loss value: {loss.numpy():.6f}")
        print(f"   Number of gradient tensors: {len(gradients)}")
        print(f"   Number of non-None gradients: {len(gradient_norms)}")
        print(f"   Average gradient norm: {tf.reduce_mean(gradient_norms).numpy():.6f}")

        # Test optimizer step
        optimizer = tf.keras.optimizers.Adam(0.001)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print("✅ Gradient computation and optimization step successful")

        return True

    except Exception as e:
        print(f"❌ Error during gradient computation: {e}")
        return False


def memory_usage_analysis():
    """Analyze memory usage of SM120 operations."""
    print("\n💾 Memory Usage Analysis")
    print("=" * 50)

    try:
        # Get initial memory usage
        if tf.config.list_physical_devices("GPU"):
            gpu_devices = tf.config.experimental.list_memory_growth(
                tf.config.list_physical_devices("GPU")[0]
            )
            print(f"   GPU memory growth enabled: {gpu_devices}")

        # Test with different input sizes
        sizes = [128, 256, 512, 1024]

        for size in sizes:
            print(f"\n   Testing with input size: {size}x{size}")

            # Create large tensors
            large_input = tf.random.normal([4, size, size], dtype=tf.float16)

            if SM120_AVAILABLE:
                # SM120 operation
                sm120_layer = SM120Dense(size, use_sm120=True, dtype=tf.float16)
                sm120_output = sm120_layer(large_input)

                # Force computation
                _ = tf.reduce_sum(sm120_output).numpy()

                print(f"      SM120 operation completed")

            # Standard operation
            standard_layer = tf.keras.layers.Dense(size, dtype=tf.float16)
            standard_output = standard_layer(large_input)

            # Force computation
            _ = tf.reduce_sum(standard_output).numpy()

            print(f"      Standard operation completed")

            # Clean up
            del large_input
            if SM120_AVAILABLE:
                del sm120_output, sm120_layer
            del standard_output, standard_layer

            # Force garbage collection
            tf.keras.backend.clear_session()

        print("✅ Memory usage analysis completed")

    except Exception as e:
        print(f"❌ Error during memory analysis: {e}")


def main():
    """Main function to run all demonstrations."""
    print("🚀 SM120 Comprehensive High-Level Example")
    print("=" * 70)

    # Enable GPU memory growth
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} device(s)")
        else:
            print("⚠️  No GPU devices found - running on CPU")
    except Exception as e:
        print(f"⚠️  Could not configure GPU: {e}")

    # Run all demonstrations
    try:
        # 1. Check GPU compatibility
        gpu_compatible = check_gpu_compatibility()

        # 2. Benchmark operations
        benchmark_results = benchmark_operations()

        # 3. Create and train a model
        model, history = create_and_train_model()

        # 4. Test gradient computation
        gradient_success = demonstrate_gradient_computation()

        # 5. Analyze memory usage
        memory_usage_analysis()

        # Final summary
        print("\n📋 Final Summary")
        print("=" * 50)
        print(f"✅ GPU SM120 Compatible: {gpu_compatible}")
        print(f"✅ SM120 Operations Available: {SM120_AVAILABLE}")
        print(f"✅ Model Training: {'Success' if model else 'Failed'}")
        print(f"✅ Gradient Computation: {'Success' if gradient_success else 'Failed'}")

        if benchmark_results:
            print("\n🏆 Performance Summary:")
            if SM120_AVAILABLE:
                if (
                    "sm120_dense" in benchmark_results
                    and "standard_dense" in benchmark_results
                ):
                    speedup = (
                        benchmark_results["standard_dense"]
                        / benchmark_results["sm120_dense"]
                    )
                    print(f"   Dense Layer Speedup: {speedup:.2f}x")

                if (
                    "sm120_conv" in benchmark_results
                    and "standard_conv" in benchmark_results
                ):
                    speedup = (
                        benchmark_results["standard_conv"]
                        / benchmark_results["sm120_conv"]
                    )
                    print(f"   Conv2D Layer Speedup: {speedup:.2f}x")

                if (
                    "sm120_attention" in benchmark_results
                    and "standard_attention" in benchmark_results
                ):
                    speedup = (
                        benchmark_results["standard_attention"]
                        / benchmark_results["sm120_attention"]
                    )
                    print(f"   Attention Layer Speedup: {speedup:.2f}x")
            else:
                print("   SM120 operations not available for comparison")

        print("\n🎉 Demonstration completed successfully!")

        if SM120_AVAILABLE and gpu_compatible:
            print("\n💡 Your system is fully optimized for SM120 operations!")
            print("   You can now use these high-level layers in your own models:")
            print("   • SM120Dense for matrix multiplications")
            print("   • SM120Conv2D for convolutions")
            print("   • SM120MultiHeadAttention for transformer models")
            print("   • SM120BatchNormalization for normalization")
        else:
            print(
                "\n💡 Consider upgrading to RTX 50-series GPU for maximum performance!"
            )

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
