loaded_model = tf.keras.models.load_model('model.h5')
result=loaded_model.predict(x_test)
print(result.shape)