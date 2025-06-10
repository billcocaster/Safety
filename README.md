scs# Kullanıcı Giriş Güvenlik Risk Değerlendirme Sistemi

Bu proje, kullanıcıların sisteme giriş yapmaya çalıştığı anda elde edilen verilerle anlık risk değerlendirmesi yapan bir yapay zeka modeli geliştirmeyi amaçlamaktadır.
<img width="1705" alt="screenshot" src="https://github.com/user-attachments/assets/fbeafdb7-0b2c-488f-9db6-c596faef40d6" />
<img width="1487" alt="screenshot2" src="https://github.com/user-attachments/assets/3d05fcd1-2dde-4817-989b-e01786d53bb7" />

## Proje Yapısı

```
Safety/
├── data/
│   ├── mock_data_generator.py      # Mock veri üretici
│   ├── risk_scorer.py              # Risk skoru hesaplama
│   └── data_processor.py           # Veri işleme
├── models/
│   ├── deep_learning_model.py      # Derin öğrenme modeli
│   └── model_trainer.py            # Model eğitimi
├── utils/
│   ├── config.py                   # Konfigürasyon
│   └── visualization.py            # Görselleştirme
├── notebooks/
│   └── risk_analysis.ipynb         # Analiz notebook'u
├── tests/
│   └── test_risk_model.py          # Test dosyaları
├── requirements.txt                 # Gerekli kütüphaneler
└── README.md                       # Bu dosya
```

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Jupyter notebook'u başlatın:

```bash
jupyter notebook
```

## Özellikler

- **Mock Veri Üretimi**: Gerçekçi kullanıcı giriş verileri
- **Risk Skoru Hesaplama**: 0-100 arası risk puanlama
- **Derin Öğrenme Modeli**: LSTM tabanlı risk tahmini
- **Esnek Ağırlık Sistemi**: Sütun bazlı risk ağırlıkları
- **Kapsamlı Analiz**: Detaylı raporlama ve görselleştirme

## Kullanım

Detaylı kullanım kılavuzu için `notebooks/risk_analysis.ipynb` dosyasını inceleyin.

## Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
