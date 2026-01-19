"""TensorFlow setup utilities for training."""

import logging
import os

# Configure TensorFlow environment variables before importing TF to silence low-level warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow.keras import backend as K

_TF_SETUP_DONE = False

def _enable_memory_growth():
    # This won't be applicable on Mac unless you have NVIDIA GPU or Metal API (for Apple Silicon).
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for GPU {gpu}")
            except RuntimeError as exc:
                print(f"Failed to enable memory growth for GPU {gpu}: {exc}")
    else:
        print("No GPU available for memory growth settings.")


def _log_device_details():
    print("Available devices:")
    for device in tf.config.list_logical_devices():
        print(f"  - {device}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nRunning on GPU ({len(gpus)} available):")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu}")
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                for key, value in gpu_details.items():
                    print(f"    {key}: {value}")
            except Exception:
                print("    No additional GPU details available.")
    else:
        print("\nRunning on CPU.")

    # Log logical GPUs (useful for multi-GPU setups)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(f"\nLogical GPUs Available: {len(logical_gpus)}")
    for i, lgpu in enumerate(logical_gpus):
        print(f"Logical GPU {i}: {lgpu}")


def _configure_tf_logs():
    tf.debugging.set_log_device_placement(True)
    tf.get_logger().setLevel('ERROR')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs


def _reset_tf_session():
    tf.keras.backend.clear_session()
    print("\nTensorFlow Build Details:")
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))
    if tf.test.is_built_with_cuda():
        print("CUDA version:", tf.__version__)
    else:
        print("TensorFlow is not built with CUDA.")


def initialize_tf():
    global _TF_SETUP_DONE
    if _TF_SETUP_DONE:
        return
    _TF_SETUP_DONE = True

    _enable_memory_growth()  # Enable memory growth for GPUs before initializing TensorFlow
    _log_device_details()
    _configure_tf_logs()
    _reset_tf_session()

    # Additional Mac-specific checks (if using Metal API for Apple Silicon)
    if tf.config.list_physical_devices('GPU'):
        if not tf.test.is_built_with_cuda():
            # If TensorFlow is built for Metal (Apple Silicon) but not CUDA, it indicates Metal backend is used
            print("\nUsing Metal API for Apple Silicon (if applicable).")
        else:
            print("\nCUDA-compatible GPU detected, using NVIDIA GPU.")


def disable_xla():
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'


initialize_tf()

try:
    # TensorFlow configuration for stability and performance
    # DISABLE eager execution for better performance with data pipelines
    tf.config.run_functions_eagerly(False)
    tf.config.experimental.enable_mixed_precision_graph_rewrite(False)

    # Configure memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("TensorFlow GPU memory growth enabled for %d GPU(s)", len(gpus))

    # Additional performance optimizations
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)

    logging.info(
        "TensorFlow %s configured: eager_execution=False, memory_growth=True",
        tf.__version__,
    )

    # Disable Keras progress bars globally to keep console output clean
    try:
        tf.keras.utils.disable_interactive_logging()
        logging.debug("TensorFlow interactive logging disabled (no progress bars).")
    except AttributeError:
        logging.debug("TensorFlow interactive logging disable not available in this version.")

except Exception as exc:
    logging.info("TensorFlow initialization warning: %s", exc)

__all__ = ["tf", "K", "initialize_tf", "disable_xla"]
