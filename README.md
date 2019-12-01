# EIP-Session-3

# Final Validation accuracy for Base Network
  82.60
  
# Final Validation accuracy for Modefied Network with depthwise separable convolution 
  83.04
  
  
# Model definition 
      model = Sequential()
      model.add(SeparableConv2D(32, 3, 3, border_mode='same', input_shape=(32, 32, 3))) #I/P:32  O/P:32 RF:3
      model.add(Activation('relu'))
      model.add(SeparableConv2D(64, 3)) #I/P:32  O/P:30 RF:5
      model.add(Activation('relu'))
      model.add(SeparableConv2D(128, 3)) #I/P:30  O/P:28 RF:7
      model.add(SeparableConv2D(256, 3,padding='same')) #I/P:28  O/P:28 RF:9
      model.add(Activation('relu'))

      model.add(BatchNormalization())
      model.add(MaxPooling2D(pool_size=(2, 2)))#I/P:28  O/P:14  RF:11

      model.add(Convolution2D(32, 1,1)) #I/P:14  O/P:14  RF:11
      model.add(SeparableConv2D(64, 3)) #I/P:14  O/P:12  RF:15
      model.add(Dropout(0.1))
      model.add(Activation('relu'))
      model.add(SeparableConv2D(128, 3)) #I/P:12  O/P:10  RF:19
      model.add(SeparableConv2D(128, 3,padding='same')) #I/P:10  O/P:10  RF:23
      model.add(Activation('relu'))

      model.add(BatchNormalization())
      model.add(Dropout(0.3))

      model.add(Convolution2D(32, 1,1)) #I/P:10  O/P:10  RF:23

      model.add(SeparableConv2D(64, 3)) #I/P:10  O/P:8  RF:27
      model.add(Activation('relu'))
      model.add(SeparableConv2D(64, 3)) #I/P:8  O/P:6  RF:31
      model.add(Activation('relu'))


      model.add(Convolution2D(10, 1, 1))


      model.add(GlobalAveragePooling2D()) 

      model.add(Activation('softmax'))
      # Compile the model
      opt = optimizers.Adam(lr=0.001);
      model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



# 50 epoch logs

    Epoch 1/50
    390/390 [==============================] - 35s 89ms/step - loss: 1.8659 - acc: 0.3021 - val_loss: 1.7764 - val_acc: 0.3538
    Epoch 2/50
    390/390 [==============================] - 29s 73ms/step - loss: 1.4419 - acc: 0.4739 - val_loss: 1.6871 - val_acc: 0.4114
    Epoch 3/50
    390/390 [==============================] - 29s 75ms/step - loss: 1.2209 - acc: 0.5625 - val_loss: 1.1653 - val_acc: 0.5781
    Epoch 4/50
    390/390 [==============================] - 29s 75ms/step - loss: 1.0670 - acc: 0.6186 - val_loss: 1.0139 - val_acc: 0.6390
    Epoch 5/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.9691 - acc: 0.6563 - val_loss: 1.0158 - val_acc: 0.6429
    Epoch 6/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.8925 - acc: 0.6854 - val_loss: 0.9467 - val_acc: 0.6657
    Epoch 7/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.8283 - acc: 0.7105 - val_loss: 0.9419 - val_acc: 0.6732
    Epoch 8/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.7712 - acc: 0.7284 - val_loss: 0.7808 - val_acc: 0.7314
    Epoch 9/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.7343 - acc: 0.7436 - val_loss: 0.7777 - val_acc: 0.7319
    Epoch 10/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.7044 - acc: 0.7534 - val_loss: 0.8690 - val_acc: 0.7081
    Epoch 11/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.6811 - acc: 0.7626 - val_loss: 0.8984 - val_acc: 0.6909
    Epoch 12/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.6531 - acc: 0.7708 - val_loss: 0.7062 - val_acc: 0.7559
    Epoch 13/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.6320 - acc: 0.7798 - val_loss: 0.7232 - val_acc: 0.7551
    Epoch 14/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.6145 - acc: 0.7859 - val_loss: 0.6860 - val_acc: 0.7665
    Epoch 15/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.5962 - acc: 0.7912 - val_loss: 0.6499 - val_acc: 0.7799
    Epoch 16/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.5789 - acc: 0.7982 - val_loss: 0.7067 - val_acc: 0.7541
    Epoch 17/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.5612 - acc: 0.8015 - val_loss: 0.6315 - val_acc: 0.7836
    Epoch 18/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.5542 - acc: 0.8062 - val_loss: 0.6294 - val_acc: 0.7814
    Epoch 19/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.5364 - acc: 0.8128 - val_loss: 0.6438 - val_acc: 0.7826
    Epoch 20/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.5270 - acc: 0.8163 - val_loss: 0.6156 - val_acc: 0.7927
    Epoch 21/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.5105 - acc: 0.8220 - val_loss: 0.5810 - val_acc: 0.8050
    Epoch 22/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.5001 - acc: 0.8250 - val_loss: 0.5860 - val_acc: 0.8081
    Epoch 23/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4920 - acc: 0.8271 - val_loss: 0.6772 - val_acc: 0.7812
    Epoch 24/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4810 - acc: 0.8314 - val_loss: 0.5923 - val_acc: 0.7962
    Epoch 25/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.4739 - acc: 0.8355 - val_loss: 0.5976 - val_acc: 0.7986
    Epoch 26/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4627 - acc: 0.8380 - val_loss: 0.5615 - val_acc: 0.8091
    Epoch 27/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4566 - acc: 0.8406 - val_loss: 0.5973 - val_acc: 0.7983
    Epoch 28/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4488 - acc: 0.8425 - val_loss: 0.5614 - val_acc: 0.8078
    Epoch 29/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.4369 - acc: 0.8470 - val_loss: 0.5550 - val_acc: 0.8153
    Epoch 30/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4301 - acc: 0.8478 - val_loss: 0.5726 - val_acc: 0.8055
    Epoch 31/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4257 - acc: 0.8527 - val_loss: 0.5585 - val_acc: 0.8166
    Epoch 32/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4234 - acc: 0.8509 - val_loss: 0.5750 - val_acc: 0.8112
    Epoch 33/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.4124 - acc: 0.8539 - val_loss: 0.5602 - val_acc: 0.8163
    Epoch 34/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4074 - acc: 0.8574 - val_loss: 0.6009 - val_acc: 0.8024
    Epoch 35/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.4010 - acc: 0.8593 - val_loss: 0.5257 - val_acc: 0.8200
    Epoch 36/50
    390/390 [==============================] - 29s 75ms/step - loss: 0.3922 - acc: 0.8626 - val_loss: 0.5575 - val_acc: 0.8161
    Epoch 37/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3908 - acc: 0.8630 - val_loss: 0.5299 - val_acc: 0.8208
    Epoch 38/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3798 - acc: 0.8646 - val_loss: 0.5417 - val_acc: 0.8205
    Epoch 39/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3777 - acc: 0.8679 - val_loss: 0.5221 - val_acc: 0.8235
    Epoch 40/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3680 - acc: 0.8685 - val_loss: 0.5467 - val_acc: 0.8256
    Epoch 41/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3659 - acc: 0.8714 - val_loss: 0.5315 - val_acc: 0.8255
    Epoch 42/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3602 - acc: 0.8736 - val_loss: 0.5065 - val_acc: 0.8343
    Epoch 43/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3567 - acc: 0.8742 - val_loss: 0.5175 - val_acc: 0.8299
    Epoch 44/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3514 - acc: 0.8762 - val_loss: 0.5725 - val_acc: 0.8201
    Epoch 45/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3499 - acc: 0.8755 - val_loss: 0.5311 - val_acc: 0.8257
    Epoch 46/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3416 - acc: 0.8805 - val_loss: 0.5186 - val_acc: 0.8287
    Epoch 47/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3379 - acc: 0.8804 - val_loss: 0.5330 - val_acc: 0.8256
    Epoch 48/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3356 - acc: 0.8812 - val_loss: 0.5550 - val_acc: 0.8213
    Epoch 49/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3341 - acc: 0.8826 - val_loss: 0.5470 - val_acc: 0.8266
    Epoch 50/50
    390/390 [==============================] - 29s 74ms/step - loss: 0.3271 - acc: 0.8838 - val_loss: 0.5301 - val_acc: 0.8304
    Model took 1458.96 seconds to train

    Accuracy on test data is: 83.04
