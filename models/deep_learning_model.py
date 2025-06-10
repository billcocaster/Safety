"""
Derin öğrenme modeli modülü
LSTM tabanlı risk skoru tahmin modeli
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import pickle
import os
from utils.config import *

class RiskPredictionModel:
    """
    Kullanıcı giriş risk skoru tahmin modeli
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        Args:
            model_config: Model konfigürasyonu
        """
        self.config = model_config if model_config else MODEL_CONFIG.copy()
        self.model = None
        self.history = None
        self.feature_dim = None
        self.user_embeddings = {}
        
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        LSTM tabanlı model oluşturur
        
        Args:
            input_shape: Giriş verisi şekli (sequence_length, feature_dim)
            
        Returns:
            Model: Oluşturulan model
        """
        print("LSTM modeli oluşturuluyor...")
        
        self.feature_dim = input_shape[1]
        
        model = Sequential([
            # İlk LSTM katmanı
            LSTM(
                units=self.config['lstm_units'],
                return_sequences=True,
                input_shape=input_shape,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate'],
                kernel_regularizer=l2(0.01)
            ),
            BatchNormalization(),
            
            # İkinci LSTM katmanı
            LSTM(
                units=self.config['lstm_units'] // 2,
                return_sequences=False,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate'],
                kernel_regularizer=l2(0.01)
            ),
            BatchNormalization(),
            
            # Dense katmanları
            Dense(
                units=self.config['lstm_units'] // 4,
                activation='relu',
                kernel_regularizer=l2(0.01)
            ),
            Dropout(self.config['dropout_rate']),
            
            Dense(
                units=self.config['lstm_units'] // 8,
                activation='relu',
                kernel_regularizer=l2(0.01)
            ),
            Dropout(self.config['dropout_rate']),
            
            # Çıkış katmanı
            Dense(1, activation='linear')  # Risk skoru için linear aktivasyon
        ])
        
        # Model derleme
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error
        )
        
        print("LSTM modeli oluşturuldu")
        model.summary()
        
        return model
    
    def build_hybrid_model(self, sequence_input_shape: Tuple[int, int], 
                          user_embedding_dim: int = 32) -> Model:
        """
        Hibrit model oluşturur (LSTM + Kullanıcı Embedding)
        
        Args:
            sequence_input_shape: Sekans giriş şekli
            user_embedding_dim: Kullanıcı embedding boyutu
            
        Returns:
            Model: Hibrit model
        """
        print("Hibrit model oluşturuluyor...")
        
        # Sekans girişi
        sequence_input = Input(shape=sequence_input_shape, name='sequence_input')
        
        # LSTM katmanları
        lstm_out = LSTM(
            units=self.config['lstm_units'],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate']
        )(sequence_input)
        lstm_out = BatchNormalization()(lstm_out)
        
        lstm_out = LSTM(
            units=self.config['lstm_units'] // 2,
            return_sequences=False,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate']
        )(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        
        # Kullanıcı embedding girişi
        user_input = Input(shape=(user_embedding_dim,), name='user_input')
        
        # Kullanıcı embedding işleme
        user_dense = Dense(
            units=self.config['lstm_units'] // 4,
            activation='relu'
        )(user_input)
        user_dense = Dropout(self.config['dropout_rate'])(user_dense)
        
        # Birleştirme
        combined = Concatenate()([lstm_out, user_dense])
        
        # Dense katmanları
        dense_out = Dense(
            units=self.config['lstm_units'] // 2,
            activation='relu',
            kernel_regularizer=l2(0.01)
        )(combined)
        dense_out = Dropout(self.config['dropout_rate'])(dense_out)
        
        dense_out = Dense(
            units=self.config['lstm_units'] // 4,
            activation='relu',
            kernel_regularizer=l2(0.01)
        )(dense_out)
        dense_out = Dropout(self.config['dropout_rate'])(dense_out)
        
        # Çıkış katmanı
        output = Dense(1, activation='linear', name='risk_score')(dense_out)
        
        # Model oluşturma
        model = Model(
            inputs=[sequence_input, user_input],
            outputs=output
        )
        
        # Model derleme
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print("Hibrit model oluşturuldu")
        model.summary()
        
        return model
    
    def build_tabular_model(self, input_dim: int) -> Model:
        """
        Tabular veri için Dense model oluşturur
        
        Args:
            input_dim: Giriş özellik boyutu
            
        Returns:
            Model: Dense model
        """
        print("Tabular model oluşturuluyor...")
        
        model = Sequential([
            Dense(
                units=128,
                activation='relu',
                input_shape=(input_dim,),
                kernel_regularizer=l2(0.01)
            ),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            Dense(
                units=64,
                activation='relu',
                kernel_regularizer=l2(0.01)
            ),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            Dense(
                units=32,
                activation='relu',
                kernel_regularizer=l2(0.01)
            ),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            Dense(1, activation='linear')
        ])
        
        # Model derleme
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print("Tabular model oluşturuldu")
        model.summary()
        
        return model
    
    def get_callbacks(self, model_path: str) -> list:
        """
        Model eğitimi için callback'leri oluşturur
        
        Args:
            model_path: Model kayıt yolu
            
        Returns:
            list: Callback listesi
        """
        callbacks = [
            # Erken durdurma
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Öğrenme oranı azaltma
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model kaydetme
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_type: str = 'lstm',
              user_embeddings: Optional[Dict[str, np.ndarray]] = None,
              model_path: str = 'models/risk_model.h5') -> Dict[str, Any]:
        """
        Model eğitimi
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedefleri
            X_val: Validation özellikleri
            y_val: Validation hedefleri
            model_type: Model tipi ('lstm', 'hybrid', 'tabular')
            user_embeddings: Kullanıcı embedding'leri (hibrit model için)
            model_path: Model kayıt yolu
            
        Returns:
            Dict: Eğitim sonuçları
        """
        print(f"{model_type.upper()} modeli eğitiliyor...")
        
        # Model oluşturma
        if model_type == 'lstm':
            self.model = self.build_lstm_model(X_train.shape[1:])
            train_data = X_train
            val_data = X_val
        elif model_type == 'hybrid':
            if user_embeddings is None:
                raise ValueError("Hibrit model için user_embeddings gerekli")
            
            self.model = self.build_hybrid_model(
                X_train.shape[1:],
                list(user_embeddings.values())[0].shape[0]
            )
            
            # Kullanıcı embedding'lerini hazırla
            train_user_embeddings = self._prepare_user_embeddings_for_training(
                X_train, user_embeddings
            )
            val_user_embeddings = self._prepare_user_embeddings_for_training(
                X_val, user_embeddings
            )
            
            train_data = [X_train, train_user_embeddings]
            val_data = [X_val, val_user_embeddings]
        elif model_type == 'tabular':
            self.model = self.build_tabular_model(X_train.shape[1])
            train_data = X_train
            val_data = X_val
        else:
            raise ValueError(f"Bilinmeyen model tipi: {model_type}")
        
        # Callback'leri oluştur
        callbacks = self.get_callbacks(model_path)
        
        # Model eğitimi
        self.history = self.model.fit(
            train_data,
            y_train,
            validation_data=(val_data, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Eğitim sonuçları
        results = {
            'model_type': model_type,
            'final_train_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
            'final_train_mae': self.history.history['mae'][-1],
            'final_val_mae': self.history.history['val_mae'][-1],
            'best_epoch': np.argmin(self.history.history['val_loss']) + 1
        }
        
        print(f"Model eğitimi tamamlandı. En iyi epoch: {results['best_epoch']}")
        
        return results
    
    def _prepare_user_embeddings_for_training(self, X: np.ndarray, 
                                            user_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Eğitim için kullanıcı embedding'lerini hazırlar
        
        Args:
            X: Özellik verisi
            user_embeddings: Kullanıcı embedding'leri
            
        Returns:
            np.ndarray: Hazırlanmış embedding'ler
        """
        # Bu fonksiyon, sekans verisindeki her örnek için kullanıcı embedding'ini bulur
        # Gerçek uygulamada, sekans verisinin hangi kullanıcıya ait olduğu bilgisi gerekir
        # Bu örnek için basit bir yaklaşım kullanıyoruz
        
        embedding_dim = list(user_embeddings.values())[0].shape[0]
        embeddings = np.zeros((X.shape[0], embedding_dim))
        
        # Gerçek uygulamada bu kısım kullanıcı ID'lerine göre doldurulmalı
        # Şimdilik rastgele embedding'ler kullanıyoruz
        for i in range(X.shape[0]):
            user_id = list(user_embeddings.keys())[i % len(user_embeddings)]
            embeddings[i] = user_embeddings[user_id]
        
        return embeddings
    
    def predict(self, X: np.ndarray, user_embeddings: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Risk skoru tahmini
        
        Args:
            X: Tahmin için özellikler
            user_embeddings: Kullanıcı embedding'leri (hibrit model için)
            
        Returns:
            np.ndarray: Tahmin edilen risk skorları
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş")
        
        if user_embeddings is not None:
            # Hibrit model için
            user_emb = self._prepare_user_embeddings_for_training(X, user_embeddings)
            predictions = self.model.predict([X, user_emb])
        else:
            # LSTM veya tabular model için
            predictions = self.model.predict(X)
        
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                user_embeddings: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Model değerlendirmesi
        
        Args:
            X_test: Test özellikleri
            y_test: Test hedefleri
            user_embeddings: Kullanıcı embedding'leri (hibrit model için)
            
        Returns:
            Dict: Değerlendirme metrikleri
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş")
        
        if user_embeddings is not None:
            # Hibrit model için
            user_emb = self._prepare_user_embeddings_for_training(X_test, user_embeddings)
            test_data = [X_test, user_emb]
        else:
            # LSTM veya tabular model için
            test_data = X_test
        
        # Model değerlendirmesi
        loss, mae = self.model.evaluate(test_data, y_test, verbose=0)
        
        # Tahminler
        predictions = self.predict(X_test, user_embeddings)
        
        # Ek metrikler
        mse = np.mean((y_test - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae_custom = np.mean(np.abs(y_test - predictions))
        
        # R-squared hesaplama
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        results = {
            'test_loss': loss,
            'test_mae': mae,
            'test_mse': mse,
            'test_rmse': rmse,
            'test_mae_custom': mae_custom,
            'test_r_squared': r_squared
        }
        
        print("Model değerlendirme sonuçları:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        return results
    
    def save_model(self, model_path: str, preprocessor_path: str = None):
        """
        Modeli kaydeder
        
        Args:
            model_path: Model kayıt yolu
            preprocessor_path: Ön işleme modelleri kayıt yolu
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok")
        
        # Model kaydetme
        self.model.save(model_path)
        print(f"Model kaydedildi: {model_path}")
        
        # Ön işleme modellerini kaydetme
        if preprocessor_path:
            preprocessors = {
                'feature_dim': self.feature_dim,
                'config': self.config
            }
            
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessors, f)
            
            print(f"Ön işleme modelleri kaydedildi: {preprocessor_path}")
    
    def load_model(self, model_path: str, preprocessor_path: str = None):
        """
        Modeli yükler
        
        Args:
            model_path: Model yükleme yolu
            preprocessor_path: Ön işleme modelleri yükleme yolu
        """
        # Model yükleme
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model yüklendi: {model_path}")
        
        # Ön işleme modellerini yükleme
        if preprocessor_path and os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessors = pickle.load(f)
            
            self.feature_dim = preprocessors['feature_dim']
            self.config = preprocessors['config']
            
            print(f"Ön işleme modelleri yüklendi: {preprocessor_path}")
    
    def get_feature_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """
        Özellik önem analizi (gradient-based)
        
        Args:
            X_sample: Örnek veri
            
        Returns:
            np.ndarray: Özellik önem skorları
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş")
        
        # Gradient hesaplama
        with tf.GradientTape() as tape:
            inputs = tf.convert_to_tensor(X_sample, dtype=tf.float32)
            tape.watch(inputs)
            predictions = self.model(inputs)
        
        gradients = tape.gradient(predictions, inputs)
        
        # Özellik önem skorları
        feature_importance = np.mean(np.abs(gradients.numpy()), axis=0)
        
        return feature_importance 