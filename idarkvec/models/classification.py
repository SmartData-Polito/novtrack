from keras import models, layers, callbacks
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import joblib


class DeepClassifier():
    def __init__(self, mname, load_model=False):
        if load_model:
            self.scaler, self.ohl_encoder = joblib.load(f"{mname}.preprocess")
            self.classifier = models.load_model(f"{mname}.h5", 
                     compile=False, custom_objects={'Functional':models.Model})
            
        else:
            self.classifier, self.scaler, self.ohl_encoder = None, None, None
        self.mname = mname
        

    def init_model(self):
        inputs = layers.Input(200, )
        hidden = layers.Dense(1024, activation='relu')(inputs)
        hidden = layers.Dense(512, activation='relu')(hidden)
        hidden = layers.Dense(256, activation='relu')(hidden)
        hidden = layers.Dense(128, activation='relu')(hidden)
        outputs = layers.Dense(self.ohl_encoder.label_encoder.classes_.shape[0], 
                                                  activation='softmax')(hidden)

        self.classifier = models.Model(inputs=inputs, outputs=outputs)
        self.classifier.compile(loss= "categorical_crossentropy" , 
                                        optimizer="adam", metrics=['accuracy'])


    def fit(self, X_train, X_val, y_train, y_val, epochs=300, with_weights=True, save=True):
        self.ohl_encoder = OHLEncoder(y_train)
        y_train_OH, weights = self.ohl_encoder.transform(y_target=y_train, 
                                                         with_weights=True)
        y_val_OH = self.ohl_encoder.transform(y_val)

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        if save:
            joblib.dump((self.scaler, self.ohl_encoder), f"{self.mname}.preprocess")
        self.init_model()
        if save:
            saver = callbacks.ModelCheckpoint(f"{self.mname}.h5", 
                                            monitor="val_loss", 
                                            verbose=1, 
                                            save_best_only=True,
                                            mode='min')
        X_train_s = self.scaler.transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        
        if not with_weights: weights = None
        if save:
            history = self.classifier.fit(x = self.scaler.transform(X_train_s), 
                                        y = y_train_OH, 
                                        batch_size=512, epochs=epochs, 
                                        validation_data = (X_val_s, y_val_OH), 
                                        sample_weight=weights, shuffle=True, 
                                        callbacks=[saver])
        else:
            history = self.classifier.fit(x = self.scaler.transform(X_train_s), 
                            y = y_train_OH, 
                            batch_size=512, epochs=epochs, 
                            validation_data = (X_val_s, y_val_OH), 
                            sample_weight=weights, shuffle=True)

        return history


    def predict(self, X_test):
        y_pred = self.classifier(self.scaler.transform(X_test))
        y_pred = np.argmax(y_pred, axis=1)

        return self.ohl_encoder.inverse_transform(y_pred)



class OHLEncoder():
    def __init__(self, y_train):
        self.label_encoder, self.onehot_encoder = None, None
        self.fit(y_train)
        
    def fit(self, y_train):
        # Fitting encoders
        unique_y = np.unique(y_train)
        self.label_encoder = LabelEncoder().fit(unique_y)
        self.onehot_encoder = OneHotEncoder(sparse=False).fit(
                         self.label_encoder.transform(unique_y).reshape(-1, 1))
        
    def transform(self, y_target, with_weights=False):
        # Train OneHotEncoding
        encoded = self.label_encoder.transform(y_target.reshape(-1, 1))
        y_OH = self.onehot_encoder.transform(encoded.reshape(-1, 1))
        # Balancing weights
        if with_weights:
            weights = compute_sample_weight(class_weight='balanced', y=y_OH)        
            
            return y_OH, weights
        
        else: return y_OH

    def inverse_transform(self, y_target):
        return self.label_encoder.inverse_transform(y_target)