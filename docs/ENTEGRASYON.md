# YuE2 - UI: Model Entegrasyon Belgesi

Bu belge, YuE2 müzik üretim modelinin YuE-UI tabanlı bir arayüze nasıl entegre edildiğini, sistemin nasıl
çalıştığını, yol boyunca karşılaşılan sorunları ve modelin neden 24 GB'lık bir NVIDIA GPU istediğini anlatır.
Kurulum ve kullanım komutları için [README.md](../README.md), Colab için
[colab/COLAB_YOL_HARITASI.md](../colab/COLAB_YOL_HARITASI.md) belgesine bakın.

---

## 1. Özet

| | |
|---|---|
| Model | YuE2-3B (`m-a-p/YuE2-3B`) + YuE2-VAE (`m-a-p/YuE2-Vae`), Python paketi `yue2-infer 0.1.5` |
| Arayüz | Gradio 5; YuE-UI'ın (joeljuvel/YuE-UI) prompt akışı üzerine YuE2 için yeniden yazıldı |
| Mimari | İki ayrı süreç: arayüz (port 7860) ↔ HTTP ↔ model sunucusu (port 7861) |
| Özellikler | Etiket + sözden şarkı, MP3'ten cover, MP3 çıktı, ilerleme/iptal, geçmiş, hızlı sentez |
| Platformlar | Apple Silicon (MPS, test edildi), NVIDIA CUDA / Colab (hazırlandı, test edilmedi) |
| Gizlilik (lokal) | Ağırlıklar yerelde, sunucu çevrimdışı modda, servisler yalnızca `127.0.0.1` adresini dinliyor |

---

## 2. Başlangıç noktası ve temel karar

Elde iki şey vardı:
1. **`yue2_infer-0.1.5-py3-none-any.whl`**: YuE2'yi çalıştıran Python kodu (66 KB). Ağırlık içermez; ağırlıklar
   Hugging Face'te ayrı tutulur (7.3 GB + 0.5 GB).
2. **YuE-UI** (github.com/joeljuvel/YuE-UI, son commit `c98cd6e`, Nisan 2025).

**YuE-UI doğrudan kullanılamadı.** YuE-UI ilk YuE modeli (v1) için yazılmış:
- 7B'lik stage-1 + 1B'lik stage-2 olmak üzere iki model ve xcodec ses kodlayıcısı kullanıyor.
- Sadece CUDA'da çalışan `exllamav2` ve `flash-attn` kütüphanelerine bağlı.

YuE2 ise tek modelli, farklı bir mimari. Bu yüzden ne YuE2 ile ne de Apple Silicon üzerinde çalışabiliyor.

**Karar:** YuE-UI deposu referans olarak klonlandı ve YuE2 için yeni bir arayüz yazıldı. YuE-UI'dan şunlar
alındı: 200 etiketlik tür/stil listesi (`top_200_tags.json`), `[verse]`/`[chorus]` söz formatı, örnek şarkı ve
proje kaydet/yükle formatı. YuE-UI'ın parça parça üretip devam etme, zaman çizelgesi ve ses dosyasıyla
yönlendirme özelliklerinin YuE2'de karşılığı olmadığı için bunlar alınmadı. Daha sonra YuE-UI'ın kullanılmayan
v1 kodu, varlıkları ve git geçmişi projeden çıkarıldı. Proje artık bağımsız bir depo.

---

## 3. Mimari

```
 Tarayıcı ──► Gradio arayüzü (source/yue2_ui.py, :7860)        ortam: yue-ui/venv (torch yok)
                    │  HTTP (JSON + dosya)
                    ▼
              Model sunucusu (server/yue2_server.py, :7861)     ortam: yue_ortam (torch, yue2)
                ├─ YuE2Pipeline (yue2 paketi) ── models/YuE2-3B, models/YuE2-Vae
                ├─ cover.py (Demucs + librosa) ── models/demucs
                └─ outputs/<iş-id>/  (audio.mp3, audio.flac, score.abc, job.json ...)
```

**Neden iki süreç?**
- **Ortam ayrımı.** yue2 belirli paket sürümlerini sabitliyor (`torch==2.10.0`, `transformers==4.57.6`,
  `numpy==2.2.6`). Gradio'nun bağımlılıkları bu sürümlerle çakışmasın diye arayüz kendi sanal ortamında çalışıyor.
- **Bağımsız yaşam döngüsü.** Arayüz model olmadan da açılabiliyor. Model sunucusu ayrıca başlatılıyor, arayüz
  5 saniyede bir sunucuyu kontrol edip kendiliğinden bağlanıyor.
- **Taşınabilirlik.** Aynı arayüz farklı makinedeki bir sunucuya da URL ile bağlanabiliyor.

### Sunucu API'si

| Uç nokta | İşlev |
|---|---|
| `GET /health` | Mod, cihaz, modelin bellekte olup olmadığı, kuyruk durumu |
| `POST /jobs` | Üretim işi başlatır (stil, sözler, CoT, seed, CFG, ABC, örnekleme, `ode_steps`) |
| `GET /jobs`, `GET /jobs/{id}` | İş listesi ve tek iş durumu (aşama, token sayıları, hata, sonuç) |
| `POST /jobs/{id}/cancel` | İptal eder; yue2'nin `cancelled` geri çağrısıyla aşama içinde durur |
| `GET /jobs/{id}/audio` | MP3 döndürür; eski FLAC işlerini gerektiğinde MP3'e çevirir |
| `POST /uploads`, `POST /analyze` | Referans şarkı yükleme ve ABC notaya çevirme (cover) |

İşler tek bir çalışan kuyruğunda sırayla işleniyor. Resmi runtime aynı anda tek iş destekliyor
(`concurrency=1`). Her iş `outputs/<iş-id>/job.json` dosyasına yazıldığı için sunucu yeniden başlasa da geçmiş korunuyor.

---

## 4. YuE2 bir şarkıyı nasıl üretiyor

Sunucu, yue2'nin `pipe(...)` çağrısını aşama aşama çalıştırıyor. Böylece arayüz ilerlemeyi gösterebiliyor ve
iptal her aşamada çalışıyor.

| Aşama | Ne olur | M4 ölçümü (49.5 sn şarkı) |
|---|---|---|
| 1. Nota planı (ABC) | Model önce akorlu bir ABC nota yazar: tempo, ton, ölçüler, bölümler | 32 sn, 488 token |
| 2. Şarkı token'ları | Nota + sözler + stil koşulunda ses token'ları üretilir (~25 token/sn ses) | 91 sn, 1238 token |
| 3. Ses sentezi (NAR) | Aynı model, flow-matching ile token'lardan sürekli ses latent'leri üretir (32 adım) | 165 sn |
| 4. Ses çözme (VAE) | Latent'ler 48 kHz stereo dalgaya çevrilir | 12 sn |
| Model yükleme | Sadece model bellekte değilken | 11 sn |

**CoT modları:**
- `full`: akorlu ABC.
- `melody`: akorsuz ABC.
- `off`: nota yok, doğrudan ses token'ları.

Dışarıdan bir ABC verilirse 1. aşama atlanır ve model bu notayı takip eder. Cover özelliği tam olarak bunu kullanıyor.

---

## 5. Entegrasyon adımları

1. **Ortamlar.** `yue_ortam` (Python 3.12, yue2 wheel kurulu) model için kullanıldı. `yue-ui/venv` arayüz için
   ayrıca oluşturuldu (Gradio 5.50).
2. **Arayüz.** Önce modelsiz ayağa kaldırıldı. Model sunucusu `--mock` modunda deneme tonu üreterek arayüzün
   baştan sona akışı doğrulandı: üretim, ilerleme, iptal, geçmiş.
3. **Ağırlıklar.** `yueModel/models/` altına bir kez indirildi. Sunucu `--offline` ve `HF_HUB_OFFLINE=1` ile
   başlatılıyor, internete hiç çıkmıyor. Yükleme sırasında `weights_manifest.json`'daki SHA-256 imzası doğrulanıyor.
4. **İlk gerçek üretim.** Mac M4 MPS üzerinde 49.5 saniyelik şarkı 5 dk 12 sn'de üretildi.
5. **Cover.** MP3 → ABC nota dönüşümü eklendi (bkz. bölüm 6).
6. **MP3 çıktı.** libsndfile 1.2.2'nin yerleşik MP3 kodlayıcısı kullanıldı; ffmpeg gerekmiyor (~320 kbps).
7. **Hızlı sentez.** İş başına `ode_steps` seçeneği eklendi: 32 yerine 16 adım. Ölçülen etkisi: sentez sesin her
   saniyesi için 3.3 sn'den 2.0 sn'ye, toplam süre ~6.3×'ten ~4.3× gerçek zamana indi.
8. **Bellek boşaltma.** Kuyruk boşalınca ağırlıklar bellekten atılıyor; sonraki işte ~11 sn'de yükleniyor. Bunu
   kapatmak için `--keep-loaded` kullanılır; Colab'da bu varsayılan olarak açık.
9. **Kalıcı servisler.** Servisler, editör kapansa da çalışmaya devam etsin diye `nohup` ile başlatıldı.
   Loglar `yue-ui/logs/` altında.
10. **Colab paketi.** `colab/` klasöründe notebook, zip paketi ve yol haritası hazırlandı. CUDA'da vokal ayırma
    GPU'da çalışıyor; FP8 seçeneği eklendi ama varsayılan olarak kapalı.

---

## 6. Cover (MP3'ten yeni şarkı)

YuE2'nin girdileri stil metni, sözler ve isteğe bağlı bir ABC notadan oluşuyor; **ses dosyası girdi olarak
alınamıyor.** Pakette sesi model token'larına çeviren bir bileşen yok. VAE'nin encoder'ı ise üretim aşamasına
bağlı değil. Bu yüzden cover, referans şarkının **notaya çevrilmesiyle** yapılıyor (`server/cover.py`):

1. **Demucs (htdemucs)** ile vokal ve enstrümanlar ayrılır.
2. **librosa** ile tempo ve ton (Krumhansl profilleri) bulunur, enstrümanlardan her ölçünün akoru çıkarılır.
3. Vokal melodisi **pYIN** ile 1/16'lık notalara oturtulur. Kayıtta vokal yoksa melodi enstrümandan alınıp
   oktav kaydırılarak vokal aralığına taşınır. Kaymalar ve oktav hataları temizlenir.
4. Sonuç, modelin kendi yazdığı ABC formatına birebir çevrilir: `L:1/32`, `V: Vocal`/`V: Ins`, `% bölüm`, vokal
   satırında akorlar.

**Korunanlar:** melodi, akorlar, tempo, bölüm yapısı. **Korunmayanlar:** orijinal sesin tınısı, aranjmanı, vokali.

**Doğrulama:** Model tarafından üretilmiş ve notası bilinen bir şarkı analiz edildi. Ton (Gm) ve akorlar
(Gm/Eb/Cm/D) doğru çıktı. Tempo 4/3 kat yanlış tahmin edildi; bu nedenle arayüze elle BPM girme alanı eklendi.

---

## 7. Karşılaşılan sorunlar ve çözümler

| Sorun | Sebep | Çözüm |
|---|---|---|
| YuE-UI çalışmıyor | YuE v1 + CUDA'ya özel kütüphaneler | YuE2 için yeni arayüz, YuE-UI varlıkları yeniden kullanıldı |
| Demucs yüklenmiyor | PyTorch 2.6+ `torch.load(weights_only=True)` | Güvenlik kontrolü kapatılmadı; checkpoint'in ihtiyaç duyduğu 4 tipe izin verildi |
| `openunmix` torchaudio istiyor | İçe aktarma bağımlılığı | torch 2.10 ile eşleşen `torchaudio==2.10.0`; yue2'nin sabit sürümleri korundu |
| 4 dakikalık şarkı iki katına çıktı | Vokalsiz kayıtta bölüm ayırıcı aynı ölçüleri hem intro hem outro yazdı | Hata düzeltildi |
| Saz kaydında melodi boş | Kayıt enstrümantal, vokal kanalı sessiz | Melodi enstrüman kanalından alınıyor, vokal aralığına taşınıyor |
| Üretim çok yavaş, Mac ısınıyor | MPS'te CUDA hızlandırmaları yok; uzun bağlam; 7.8 GB swap | Hızlı sentez, işten sonra bellek boşaltma; sonunda modeller Colab'a (L4) taşındı, Mac'te yalnızca mock sunucu |
| Butona basınca bir şey olmuyor | Sunucu güncellendi, tarayıcıdaki sayfa eski kaldı | Sayfayı Cmd+Shift+R ile tamamen yenilemek |
| Arayüz modeli "hazır" gösteriyordu | Dosya doğrulaması ile ağırlıkların yüklenmesi ayırt edilmiyordu | Bayrak sadece ağırlıklar gerçekten bellekteyken açık |

---

## 8. Model neden 24 GB VRAM'li bir NVIDIA GPU istiyor?

Kısa cevap: **24 GB, modelin rahat sığdığı ve resmi ayarların hedeflendiği bellek miktarı. NVIDIA ise yue2'nin
hız optimizasyonlarının yalnızca CUDA'da çalışmasından kaynaklanıyor.** Model 24 GB'tan az bellekte de
çalışabiliyor; bu bir alt sınır değil, resmi hedef ayar.

### 8.1 Bellek: 24 GB nereden geliyor?

GPU belleğini dolduran kalemler (`config.json` ve koddan hesaplanmış yaklaşık değerler):

| Kalem | Boyut | Açıklama |
|---|---|---|
| Model ağırlıkları | **7.3 GB** | ~3.6 milyar parametre × 2 bayt (BF16). Bunun ~1.5 GB'ı 184.704 kelimelik dev sözlüğün giriş ve çıkış katmanları |
| KV cache | **~1.2 GB** (9000 token) – **2.8 GB** (tam bağlam) | Üretilen her token'ın geçmişi: 28 katman × 8 KV kafası × 128 boyut × token sayısı × 2 (anahtar+değer) × 2 bayt |
| CFG ikinci dalı | **+ aynı miktar** | CFG kullanılınca (ör. `off` modu, varsayılanı 1.01) model iki dal çalıştırır; cache ikiye katlanır |
| Ses sentezi + VAE | **~1-3 GB** | Sentezde dikkat hesapları, 0.5 GB VAE ve parça parça ses çözme tamponları |
| Güvenlik payı | **2 GB** | Pipeline, ayrılan bütçeden 2 GB'ı bilerek boşta bırakıyor (`memory_budget_gib - 2`) |

**Toplam ~12-17 GB** ediyor. Resmi ayar (`memory_budget_gib=24`) uzun şarkılarda, CFG'de ve parça boyutlarında
(`vae_core_frames=1024`) rahat bir pay bırakmak için 24 GB'ı hedefliyor. Bütçe 12 GB'ın altına düşerse pipeline
ses çözmeyi daha küçük parçalarla (512) yapıyor. Yani model daha az bellekte çalışabiliyor ama daha yavaş ve
daha az payla.

### 8.2 Neden NVIDIA?

yue2'nin hız ve verimlilik özellikleri koddan görüldüğü kadarıyla CUDA'ya bağlı:

| Özellik | CUDA'da | Apple MPS'te |
|---|---|---|
| **BF16 zorunluluğu** | Pipeline BF16 desteği yoksa CUDA'da çalışmayı reddediyor | Çalışıyor ama donanım hızlandırması sınırlı |
| **CUDA graph** (`cuda_graph.py`) | Token üretim adımı önceden kaydedilip tek seferde çalıştırılıyor | Yok; her token Python'dan ayrı ayrı çalışıyor (eager) |
| **vLLM arka ucu** (`fast.py`, `[fast]` ekstrası) | Daha hızlı token üretimi | Yok |
| **FP8 hızlandırma** (`quantization.py`) | Compute capability 8.9+ gerekiyor (L4, RTX 40xx, H100) | Yok |
| **Bellek bütçesi** | `set_per_process_memory_fraction` ile sınırlanıyor | Birleşik bellek; CPU ile paylaşılıyor, dolunca swap'e taşıyor |

Mac M4'teki etkisi ölçüldü: token üretimi başlangıçta ~14 token/sn, uzun şarkıda 6.7 token/sn'ye düştü ve
toplam üretim ~4-6.5× gerçek zaman sürdü. Mac'te CPU ile GPU aynı 24 GB'ı paylaştığı için 4 dakikalık bir
denemede 7.8 GB swap kullanıldı.

### 8.3 Daha küçük GPU'lar ne durumda?

- **T4 (16 GB):** Bellek açısından sığıyor (~12-14 GB, payı az). Ama Turing mimarisi BF16'yı donanımda
  hızlandırmıyor ve FP8 desteklemiyor; yavaş çalışır. `--offload-ar` ve daha düşük `--budget` gerekebilir.
- **L4 (24 GB):** Resmi hedefle birebir uyumlu: BF16 donanım desteği, FP8 desteği (compute 8.9), uygun birim
  maliyeti. Colab için önerilen kart bu.
- **8 GB kartlar (ör. RTX 3050):** Model sunucusu `--phase-offload` ile her aşamada modelin yalnızca kullanılan
  yarısını GPU'ya koyuyor (token üretiminde AR yarısı ~4.4 GB, sentezde NAR yarısı ~2.8 GB); en yüksek kullanım
  ~6 GiB. Kuantizasyon yok, hesaplama resmi runtime'la aynı. Ampere kartlar BF16'yı destekler, FP8'i desteklemez.

---

## 9. Bilinen sınırlar ve açık işler

- Colab notebook'u henüz Colab'da çalıştırılmadı; L4 üzerindeki hız ölçülmedi.
- Arayüzdeki tahmini süreler, sunucunun aynı GPU'daki önceki üretimlerinden hesaplanır; L4'te ilk üretimden
  önce tahmin gösterilmez.
- Söz tanıma ve GPU paylaşımı (üretim ile analizin sırayla çalışması, gerekirse YuE2'nin boşaltılması) Colab'da
  denenmedi.
- Hızlı sentezin (16 adım) ve FP8'in ses kalitesine etkisi karşılaştırmalı dinlenmedi.
- Cover'da tempo tahmini bazen yanılıyor (4/3, 2 kat). Ölçü başları (downbeat) tahmin edilmiyor; 4/4 varsayılıyor.
- Model referans süresine tam uymuyor: 25 sn'lik referanstan 40 sn'lik şarkı çıktı.
- Model ilk testte sözlerin `[chorus]` bölümünü notaya hiç yazmadı; bunun seed'e özgü olup olmadığı bilinmiyor.
