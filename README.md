# 💊 Pharmaceutical Drugs and Vitamins Classification

Bu proje, Kaggle üzerindeki "Pharmaceutical Drugs and Vitamins" veri setini kullanarak, görüntü işleme ve derin öğrenme yöntemleriyle ilaç/vitamin sınıflandırması yapmaktadır.

## 🚀 Proje Hakkında
Bu çalışmada, önceden eğitilmiş (pretrained) **MobileNetV2** modeli üzerine Transfer Learning uygulanarak yüksek doğruluklu bir sınıflandırıcı oluşturulmuştur. Proje kapsamında veri görselleştirme, veri ön işleme, model eğitimi ve sonuç analizi adımları uygulanmıştır.

### 🛠 Kullanılan Teknolojiler
* **Python 3.x**
* **TensorFlow / Keras** (Model mimarisi ve eğitim)
* **Pandas & NumPy** (Veri yönetimi)
* **Matplotlib** (Görselleştirme)
* **Scikit-learn** (Veri seti bölme ve raporlama)

## 📊 Veri Seti
* Toplam 10 farklı sınıf (İlaç ve vitamin türleri).
* Görüntü boyutları: 224x224 (RGB).

## 🧠 Model Mimarisi
* **Base Model:** MobileNetV2 (ImageNet ağırlıkları kullanıldı).
* **Ek Katmanlar:** GlobalAveragePooling2D, Dense (256, ReLU), Dropout (0.2).
* **Optimizer:** Adam (Learning Rate: 0.0001).
* **Loss:** Categorical Crossentropy.

## 📈 Teknik Detaylar
* **Early Stopping:** Modelin aşırı öğrenmesini (overfitting) önlemek için `val_accuracy` takibi yapıldı.
* **Model Checkpoint:** En iyi ağırlıklar otomatik olarak kaydedildi.
* **ImageDataGenerator:** Görüntülerin modele girmeden önce MobileNetV2 standartlarına uygun şekilde ön işlemesi yapıldı.

## 💻 Nasıl Çalıştırılır?
1. Repoyu klonlayın.
2. Gerekli kütüphaneleri yükleyin: `pip install tensorflow pandas matplotlib scikit-learn`.
3. Kaggle üzerinden veri setini indirip proje dizinine ekleyin veya Kaggle Notebook üzerinde çalıştırın.
4. `transferlearning.py` dosyasını çalıştırın.

---
*Hazırlayan: İbrahim Çinğay
