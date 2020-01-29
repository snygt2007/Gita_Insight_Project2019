from Library_image_supervised_model import *

dir_litw_resized=Path("./data/resized_images_details")
dir_litw =Path("./data/Raw_data")
dir_litw_super_clean=Path("./data/super_images_cleaned")



# obtain clean data
company_list, df_image_total_brand = Get_cleaned_supervised_folder_info(dir_litw_super_clean)





print(company_list)



# split in train test using random shuffle
[X_train,y_train,X_val,y_val,X_test,y_test,X_orig_train,X_orig_val,X_orig_test]=Get_train_val_test_cleaned(company_list,df_image_total_brand,dir_litw_super_clean)





print(y_test)


# Training of models
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=2)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
#x = Dropout(0.3)(x)
predictions_layer = Dense(25, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions_layer)
for layer in base_model.layers:
    layer.trainable = True
    


model.summary()
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
y_test_1=np.array(y_test.astype(np.int64))




y_binary_train = to_categorical(y_train)
y_binary_valid = to_categorical(y_val)
y_binary_test=to_categorical(y_test)
model.fit((X_train), y_binary_train, shuffle=True, validation_data=((X_val), y_binary_valid), 
          batch_size=50, epochs=50, verbose=1, callbacks=[early_stopping])





[predict_test_results,y_test_1,y_test,y_predict] = print_predict_convert(X_test,y_test,model)

plot_confusion_mat(y_test_1,y_predict)




sys.getdefaultencoding()



feature_engg_data_inter = save_model(model)





