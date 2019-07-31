

if __name__ == "__main__":
    print("> Test tensorflow-gpu")
    import tensorflow as tf
    is_gpu_available = tf.test.is_gpu_available()
    print(">> __version__: ", tf.__version__)
    print(">> is_gpu_available:", is_gpu_available)

