# YuE2 - UI: Colab Yol Haritası

Lokaldeki kurulumun (YuE2 model sunucusu + Gradio arayüzü + MP3 cover) Google Colab'a taşınması.

## Ne değişiyor, ne aynı kalıyor

| | Lokal (Mac M4) | Colab |
|---|---|---|
| Arayüz, cover, MP3 çıktısı | aynı | aynı |
| Donanım | Apple MPS | NVIDIA L4 (CUDA) |
| Model ağırlıkları | `yueModel/models/` (bir kez indirildi) | ilk oturumda Hugging Face'ten iner ve Drive'a kaydedilir (~15 GB); sonra Drive'dan açılır |
| Python ortamları | `yue_ortam`, `sheetsage_ortam` (bir kez kuruldu) | ilk oturumda kurulur ve Drive'a kaydedilir (~10 GB); sonra Drive'dan açılır |
| Vokal ayırma | CPU | GPU (`--analysis-device auto`) |
| Söz tanıma (Whisper) | GPU (MPS), fp32 | GPU, fp16 |
| Model belleği | her üretimden sonra boşaltılır | bellekte tutulur (`--keep-loaded`); analize yer yetmezse boşaltılır |
| Tahmini süreler | M4 ölçümü | GPU'daki ilk üretimden sonra otomatik hesaplanır |
| Arayüz linki | `127.0.0.1:7860` | Colab proxy linki (yalnızca senin hesabın) |
| Veri | Mac'ten çıkmaz | MP3'ler ve şarkılar Google sunucularında işlenir |

## Adım 0: Ön koşullar (bir kez)
1. **Google AI Pro aboneliği.** Colab compute birimleri (~200 CU/ay) bu plana dahil.
2. **GPU seçimi:** colab.research.google.com adresinde *Çalışma zamanı → Çalışma zamanı türünü değiştir → L4 GPU*.
   L4 yoksa A100 seç (saatte ~3 kat fazla birim harcar). T4 de çalışır ama yavaştır.
3. **Google Drive (~25 GB boş yer):** Colab her oturumda boş bir makineyle başlar. Model ağırlıkları ve Python ortamları
   her seferinde yeniden inip kurulmasın diye notebook onları ilk oturumda Drive'daki `yue2_cache` klasörüne kaydeder
   (`DRIVE_CACHE = True`).
   Şarkıların da silinmemesini istersen `USE_DRIVE = True` yap.

## Adım 1: Paketi hazırla (Mac'te, kod her değiştiğinde)
```bash
cd ~/Desktop/yueModel/yue-ui && ./colab/make_bundle.sh
```
Bu komut `colab/yue2-colab.zip` dosyasını üretir (~100 KB). İçinde arayüz, model sunucusu, cover ve söz analizi ile
yue2 wheel'i var. Model ağırlıkları zip'te değil, Colab'da indirilir.

> İpucu: Zip'i Google Drive'ın ana klasörüne koyarsan, Drive bağlıyken notebook onu oradan kendisi alır.
> Her oturumda dosya seçmek zorunda kalmazsın.

> ⚠️ Kod değiştiyse Drive'daki eski `yue2-colab.zip`'i yenisiyle **değiştir**. Notebook Drive'da zip bulursa
> dosya sormadan onu kullanır, yani eski zip orada kalırsa arayüz de eski kalır (ör. Text to Song sekmesi görünmez).
> Aynı oturumda yeni zip yüklüyorsan önce *Çalışma zamanı → Oturumu yeniden başlat* yap ya da `/content/yue2-colab.zip`'i sil.

## Adım 2: Notebook'u aç (bir kez)
colab.research.google.com adresinde *Dosya → Notebook yükle* ile `colab/YuE2_Colab.ipynb` dosyasını seç.
Colab notebook'u Drive'ındaki "Colab Notebooks" klasörüne kaydeder; sonraki seferlerde oradan açarsın.

## Adım 3: Her oturumda hücreleri çalıştır
Hepsini birden çalıştırmak için üst menüden *Çalışma zamanı → Tümünü çalıştır* (⌘F9). Son hücre (9. Bitir)
**KAPAT** kutucuğu işaretli değilse hiçbir şeyi kapatmaz. Sizden bir şey isteyen yerler 2. hücrenin Drive izin
penceresi ve (zip Drive'da değilse) 3. hücrenin dosya seçme penceresidir.

| Hücre | Ne yapar | Süre |
|---|---|---|
| 1. GPU kontrolü | GPU'yu ve belleğini gösterir; GPU yoksa durur | saniyeler |
| 2. Drive | Drive'ı bağlar; önbellek `yue2_cache`, `USE_DRIVE = True` ise çıktılar da Drive'a | saniyeler |
| 3. Paketi yükle | Zip'i ister (ya da Drive'dan alır) ve açar | saniyeler |
| 4. Paketleri kur | yue2, torch 2.10, Demucs, librosa, Gradio; ayrı ortam (`/content/yue2_ortam`). İlk oturumda kurulur ve Drive'a yazılır, sonra Drive'dan açılır | ilk: ~3-6 dk, sonra: Drive hızına bağlı |
| 5. Ağırlıklar | YuE2-3B + VAE + Demucs + Whisper (turbo + Türkçe). İlk oturumda iner ve Drive'a yazılır, sonra Drive'dan açılır | ilk: ~5-10 dk, sonra: Drive hızına bağlı (birkaç dk) |
| 5b. SheetSage2 | Nota çıkarıcı: ayrı Python 3.11 ortamı (torch 2.8) + model + MERT-v2 (2.5 GB); hepsi 5'teki gibi önbellekli. Hata verirse arayüzde *Nota çıkarıcı: Basit analiz* seçilir | ~3-5 dk |
| 6. Model sunucusu | Port 7861; GPU belleğine göre ayar yapar | ~30 sn |
| 7. Arayüz | Port 7860; Colab linkini yazar ve arayüzü hücrenin altında da açar | ~20 sn |

İndirme ve kurulum yalnızca ilk oturumda yapılır; 4, 5 ve 5b hücreleri sonunda ne kadar sürdüklerini yazar.
4. hücredeki sürümler, zip'teki yue2 wheel'i ya da SheetSage2'nin gereksinimleri değişirse ilgili ortam kendiliğinden
yeniden kurulur ve eskisi Drive'dan silinir. Her şeyi sıfırlamak için (ör. model güncellenince) Drive'daki
`yue2_cache` klasörünü silmen yeterli.

## Adım 4: Kullanım
- Linki aç. Arayüz lokaldekiyle aynı.
- **✦ Text to Song:** şarkı açıklamasını ve sözleri (`[verse]`, `[chorus]` bölümleriyle) yaz, **🎙 Vokal** ve
  **🎚 Tür** seç, **⚡ Şarkı üret**'e bas. MP3 ve nota çıkarma gerekmez. Bu sekme için yalnızca YuE2 kullanılır;
  5b (SheetSage2) hücresi hata verse bile çalışır. *Tümünü çalıştır* hata veren hücrede durduğu için bu durumda 6 ve
  7'yi elle çalıştır. Modele giden metin sonuçtaki *Modele giden girdi* alanında görünür.
  Sonuçları karşılaştırmak için *Gelişmiş ayarlar*'da *Rastgele seed*'i kapatıp seed sabitle. Geçmiş'te ✦ ile işaretli
  üretimlerin girdileri forma geri yüklenebilir.
- **🎵 Cover:** MP3 yükle. Sözlü bir şarkıysa *Söz dili*'ni seç (Türkçe şarkıda "Türkçe") ve **✨ Sözleri tanı**'ya
  bas; sözler şarkıdaki yerleriyle gelir. Yanlış kelimeleri düzelt (bölüm etiketlerine dokunma), etiketleri seç,
  **Şarkı üret**'e bas. *Süre* 0 = tüm şarkı; hızlı deneme için bir bölüm seç.
- Üretim sürerken yeni bir şarkı tanıtılamaz ("bitince tekrar deneyin" uyarısı). İkisi GPU'ya birlikte sığmaz.
- İlk üretimden sonra arayüz, tahmini süreleri bu GPU'nun gerçek hızından hesaplar.
- **Colab sekmesini açık tut.** Pro planda arka planda çalıştırma yok; sekme kapanır ya da uzun süre boşta
  kalırsa oturum düşer.
- Üretilen MP3'leri arayüzdeki indirme butonuyla indir. Drive kullanmıyorsan oturum kapanınca silinirler.

## Adım 5: Bitirme (önemli)
**9. hücredeki KAPAT kutucuğunu** işaretleyip hücreyi çalıştır. Sunucuları kapatır ve GPU'yu bırakır. GPU bağlı
kaldığı sürece, üretim yapmasan bile compute birimi harcanır.

İlk oturumda KAPAT, önbelleğin (~25 GB) Drive'a yüklenmesi bitene kadar bekler; birkaç dakika sürebilir.
Sekmeyi o sırada kapatma: yükleme yarım kalırsa notebook bunu sonraki oturumda fark eder ve ağırlıkları ve ortamları yeniden indirip kurar.

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| "GPU yok" / L4 listede yok | Çalışma zamanı türünden L4 seç; müsait değilse A100 seç ya da biraz bekle |
| Arayüz linki açılmıyor veya boş sayfa geliyor | 7b hücresi: Gradio'nun herkese açık linki (paylaşma, 72 saat geçerli) |
| Üretimde bellek hatası (OOM) | Hızlı sentezi aç, Süre'yi kısalt. 16 GB kartlarda `--offload-ar` zaten otomatik |
| *Sözleri tanı*'da bellek hatası | Sunucu YuE2'yi gerektiğinde kendisi boşaltır; yine olursa 8. hücrenin çıktısını gönder |
| Şarkı 0:00 görünüyor, indirme "sunucu IP adresi bulunamadı" diyor | Arayüz Colab adresini bilmeden başlatılmış (eski 7. hücre). Güncel 7. hücre `GRADIO_ROOT_PATH` verir; şarkıyı hemen almak için 8b hücresi |
| *Sözleri tanı*: "Could not import module 'pipeline'" | Colab'ın torchvision'ı torch 2.10 ile uyumsuz. Güncel notebook'ta yue2 ayrı bir ortamda çalıştığı için olmaz; eski notebook'u yenisiyle değiştir |
| "Şu an şarkı üretiliyor; bitince tekrar deneyin" | Normal: üretim bitince tekrar bas |
| Text to Song sekmesi yok, arayüz eski | Drive'daki `yue2-colab.zip` eski: Adım 1'deki uyarıya bak |
| Oturum düştü | 1'den itibaren tekrar çalıştır. Drive'daki çıktılar ve model önbelleği korunur |
| 4, 5 veya 5b "Önbellek eksik" diyor | Önceki oturumda Drive yüklemesi bitmemiş; hücre yeniden indirip/kurup önbelleği yeniden yazar |
| Başka bir hata | 8. hücrenin (loglar) çıktısını gönder |

## Bilinen sınırlar
- Notebook Colab'da henüz test edilmedi; Colab erişimim yok. İlk çalıştırmada çıkan hataları 8. hücrenin
  çıktısıyla birlikte gönder.
- L4'te üretim ölçüldü: 284.7 sn'lik (~4:45) cover 282 sn'de üretildi, yani ~1× gerçek zaman (Mac M4'te ~6.5×).
  Söz tanıma hızı henüz ölçülmedi. Arayüzdeki tahminler bu ölçümlerden otomatik oluşur.
- Söz tanıma (Whisper) ve GPU paylaşımı (analizden önce YuE2'nin boşaltılması) Colab'da henüz denenmedi;
  bellek değerleri (YuE2 ~7.3 GB, Demucs + Whisper ~5-6 GB) tahmin. İlk *Sözleri tanı*'dan sonra 8. hücrenin
  çıktısını gönder.
- yue2'nin L4'te kullanılabilen FP8 hızlandırması (`--quantization fp8`) varsayılan olarak kapalı.
  Kalite etkisi denenmeden açılmamalı.
