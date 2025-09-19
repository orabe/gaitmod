#!/usr/bin/env python
# python clear_memory.py
"""
Memory cleanup script for TensorFlow/GPU training processes
"""
import gc
import os
import sys

def clear_memory():
    print("Starting memory cleanup...")
    
    # Force garbage collection
    print("Running garbage collection...")
    collected = gc.collect()
    print(f"   Collected {collected} objects")
    
    # Clear TensorFlow GPU memory if available
    try:
        import tensorflow as tf
        print("Clearing TensorFlow GPU memory...")
        
        # Clear session if exists
        if hasattr(tf.keras.backend, 'clear_session'):
            tf.keras.backend.clear_session()
            print("   Keras session cleared")
            
        # Reset default graph
        if hasattr(tf, 'reset_default_graph'):
            tf.reset_default_graph()
            print("   Default graph reset")
            
        # Clear GPU memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"   Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("   GPU memory growth enabled")
        else:
            print("   No GPUs found")
            
    except ImportError:
        print("   TensorFlow not available")
    except Exception as e:
        print(f"   TensorFlow cleanup error: {e}")
    
    # Final garbage collection
    gc.collect()
    print("Memory cleanup complete!")

if __name__ == "__main__":
    clear_memory()
