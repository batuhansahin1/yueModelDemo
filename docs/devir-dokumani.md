# YuE2 - UI: Devir Dokümanı

**Tarih:** 16 Ekim 2026
**Kimin için:** Projeyi devralan mühendis. Python ve temel web bilgisi varsayılır; müzik üretim modelleri hakkında
ön bilgi gerekmez (kısa sözlük: [bölüm 12](#12-sözlük)).

Bu belge projenin bugünkü durumunu, nasıl ayağa kaldırılacağını (Mac, Windows, Ubuntu), Colab paketinin ve notebook'un nasıl üretildiğini,
model ve arayüz modlarını ve açık işleri tek yerde toplar. Ayrıntı gereken her konuda ilgili belgeye bağlantı verilir
(belge haritası: [bölüm 11](#11-belge-haritası)).

---

## 1. Proje özeti

Proje, açık ağırlıklı müzik üretim modeli **YuE2** (`m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`) için bir model sunucusu ve iki
arayüzden oluşur. Kullanıcı iki şey yapabilir:

- **✦ Text to Song:** Açıklama, tür ve `[verse]`/`[chorus]` etiketli sözlerden sıfırdan şarkı üretir.
- **🎵 Cover:** Yüklenen bir MP3'ün melodisini ve yapısını koruyarak, seçilen türde yeni bir şarkı üretir.

YuE2 ses dinleyemez; girdileri yalnızca metindir (stil etiketleri, sözler, isteğe bağlı ABC nota). Cover bu yüzden
referans şarkıyı önce **ABC notaya** (ve isterse sözlere) çevirir, YuE2'ye bu metinleri verir. Bunun için yardımcı
modeller kullanılır: SheetSage2 (nota çıkarma), Demucs (vokal ayırma), Whisper (söz tanıma).

### Bileşenlerin durumu

| Bileşen | Durum |
|---|---|
| Model sunucusu (`server/yue2_server.py`) | Çalışıyor. Mac M4 (MPS) ve Colab L4 (CUDA) üzerinde kullanıldı |
| Gradio arayüzü (`source/yue2_ui.py`) | Çalışıyor. Colab'da kullanılan arayüz bu |
| Web arayüzü (`frontend/`) | Çalışıyor, testleri var. Yayına alınmadı; Colab paketinde yok |
| Colab | Notebook ve paketleme betiği hazır; paket güncel koddan `make_bundle.sh` ile üretilir (bkz. [6.2](#62-paket-make_bundlesh)). L4'te ~1× gerçek zaman ölçüldü (4:45'lik cover 282 sn) |
| Windows + RTX 3050 8 GB | Kurulum betikleri ve `--phase-offload` hazır. **Gerçek donanımda hiç çalıştırılmadı**. Adım adım: [4.4](#44-windows-başlangıç-sihirbazı) |
| Ubuntu (Linux + NVIDIA) | Kod Linux'u destekliyor (`.sh` betikleri, `bin/python` yolları); kurulum betiği yok, **hiç denenmedi**. Elle kurulum ve olası sorunlar: [4.5](#45-ubuntu-kurulum-ve-başlatma), [4.6](#46-ubuntu-olası-sorunlar-ve-çözümleri) |
| Herkese açık site için kuyruk (Redis) | Tasarım kararı verildi, kod yok (bkz. [bölüm 10](#10-açık-işler)) |

---

## 2. Depo ve klasör düzeni

**Git deposu yalnızca `yue-ui/` klasörüdür.** Remote: `https://github.com/batuhansahin1/yueModelDemo` (`main`;
`e050e98` 13 Ekim, `3aa72e0` 16 Ekim: bu belge ve `model-calisma-prensibi.md`). Şu belgeler henüz commit edilmedi:
`docs/gun-gun-gelistirme-plani.md`, `docs/rtx3050-windows-raporu.md`.

Betikler ağırlıkları ve Python ortamlarını `yue-ui`'nin **bir üst klasöründe** arar. Bu yüzden depo her zaman
`<kök>/yue-ui` olarak klonlanır:

```
yueModel/                          ← depo değil; yue-ui'nin yanında durması gerekenler
├── yue-ui/                        ← git deposu
│   ├── server/                    model sunucusu ve analiz kodu (yue_ortam'da çalışır)
│   │   ├── yue2_server.py         FastAPI, :7861. İş kuyruğu, /analyze, GPU paylaşımı, bellek modları
│   │   ├── cover.py               Cover analizi: "Basit analiz" (Demucs + librosa) ve SheetSage2 yolunun birleştiricisi
│   │   ├── lyrics.py              Whisper ile söz tanıma, sözleri bölümlere yerleştirme
│   │   ├── sheetsage.py           SheetSage2'yi ayrı süreçte çağırır
│   │   └── sheetsage_runner.py    sheetsage_ortam'ın Python'uyla çalışan alt süreç
│   ├── source/yue2_ui.py          Gradio arayüzü, :7860 (torch kurulu değil, modeli URL ile tanır)
│   ├── frontend/                  Web arayüzü, :7870 (stdlib Python sunucusu + derlemesiz HTML/JS)
│   ├── colab/                     Notebook, paketleme betiği, Colab yol haritası (zip burada üretilir, git dışı)
│   ├── docs/                      Belgeler
│   ├── wheels/                    yue2_infer-0.1.5 wheel'i (66 KB, ağırlık içermez)
│   ├── download_models.py         Eksik ağırlıkları indirir (Mac, Windows ve Linux'ta aynı)
│   ├── start_model_server.sh/.bat, start_ui_mac.sh, start_ui_windows.bat, setup_windows.bat
│   ├── top_200_tags.json, examples/example.json   YuE-UI'dan alınan veriler (LICENSE, NOTICE)
│   ├── venv/                      Gradio ortamı, yalnızca Mac (git dışı)
│   └── outputs/, logs/            Üretimler, yüklemeler, loglar (git dışı)
├── yue_ortam/                     Python 3.12.10 + torch 2.10 + yue2 + Demucs/Whisper bağımlılıkları
├── sheetsage_ortam/               Python 3.11 + torch 2.8 (SheetSage2)
├── .python/                       uv'nin indirdiği Python 3.11; sheetsage_ortam buna bağlı, silinirse ortam bozulur
├── models/                        Ağırlıklar (~12 GB, tablo aşağıda)
├── yedek-sozlerden-9ekim/         Geri alınan "Sözlerden" modunun tek kopyası (git'te yok, bkz. 5.7)
├── staj-konu-aciklamalari.md      Gün gün kavram notları (staj raporu taslağı)
└── yue2_infer-0.1.5-...whl        Eski kopya; depodaki betikler wheels/ altındakini kullanıyor
```

Depo dışında bir parça daha var: SheetSage2'nin temel modeli **MERT-v2** (2.4 GB) `models/` altında değil, Hugging
Face önbelleğinde durur (`~/.cache/huggingface/hub/models--m-a-p--MERT-v2-FullSong`; `HF_HOME` ile taşınabilir).

| Model | Klasör | Boyut | Ne için | Lisans |
|---|---|---|---|---|
| YuE2-3B | `models/YuE2-3B` | 7.3 GB | Şarkı üretimi | CC BY-NC 4.0 |
| YuE2-VAE | `models/YuE2-Vae` | 0.5 GB | Latent'leri sese çevirme | CC BY-NC 4.0 |
| SheetSage2 | `models/SheetSage2` | 0.2 GB | Cover: nota çıkarma | CC BY-NC 4.0 |
| MERT-v2 | HF önbelleği | 2.4 GB | SheetSage2'nin temel modeli (sabit revizyon) | |
| Demucs htdemucs | `models/demucs` | 80 MB | Cover: vokal ayırma | |
| Whisper large-v3 turbo | `models/whisper-large-v3-turbo` | 1.5 GB | Dil algılama, Türkçe dışı sözler | |
| Whisper large-v3 Turkish | `models/whisper-large-v3-turkish-general` | 2.9 GB | Türkçe sözler | Apache-2.0 |

> **Lisans uyarısı:** YuE2 ve SheetSage2 ağırlıkları ticari olmayan kullanım içindir (CC BY-NC 4.0). Herkese açık
> demo planlanırken bu göz önünde tutulmalı.

---

## 3. Mimari

```
Tarayıcı ─┬─► Gradio arayüzü   source/yue2_ui.py     :7860   Mac: yue-ui/venv · Windows/Colab: yue_ortam
          └─► Web arayüzü      frontend/server.py    :7870   Python stdlib; /api/* isteklerini aktarır
                         │  HTTP (JSON + dosya)
                         ▼
          Model sunucusu   server/yue2_server.py  :7861   yue_ortam (torch 2.10, yue2)
            ├─ YuE2Pipeline ─────────── models/YuE2-3B, models/YuE2-Vae
            ├─ cover.py ─── Demucs ──── models/demucs (+ librosa, pYIN)
            ├─ lyrics.py ── Whisper ─── models/whisper-* (bellekte aynı anda tek model)
            ├─ sheetsage.py ──alt süreç──► sheetsage_runner.py (sheetsage_ortam, torch 2.8)
            │                               └─ models/SheetSage2 + MERT-v2
            └─ outputs/<iş-id>/ , outputs/uploads/<yükleme-id>/
```

**Temel tasarım kararları ve nedenleri:**

| Karar | Neden |
|---|---|
| Arayüzler model yüklemez, model ayrı süreçte | yue2 bağımlılık sürümlerini sabitliyor (`torch==2.10.0`, `transformers==4.57.6`, `numpy==2.2.6`); Gradio'yla çakışmasın. Arayüz modelsiz açılır, sunucuyu 5 sn'de bir yoklar |
| SheetSage2 ayrı ortamda, alt süreç olarak | torch 2.8 istiyor; yue2 torch 2.10. Süreç kapanınca belleğin tamamı geri verilir |
| Tek çalışanlı iş kuyruğu | yue2'nin resmi runtime'ı aynı anda tek üretim destekliyor |
| Tek GPU kilidi (`gpu_lock`) | Üretim ve analiz birlikte 24 GB'a sığmıyor. Üretim sürerken yeni analiz **409** alır; analiz sürerken gelen üretim kuyrukta bekler |
| Her iş `outputs/<iş-id>/job.json`'a yazılır | Sunucu yeniden başlasa da geçmiş korunur; yarıda kalan işler `failed` olur |
| Tamamen çevrimdışı (`HF_HUB_OFFLINE=1`), yalnızca `127.0.0.1` | Gizlilik; ağırlıklar yerelde. Web arayüzü yalnızca izinli uç noktaları aktarır, model sunucusu dışarı açılmaz |

**Model sunucusu API'si:**

| Uç nokta | İşlev |
|---|---|
| `GET /health` | Mod (`yue2`/`mock`), cihaz, model bellekte mi, FP8 ve SheetSage2 hazır mı, kuyruk |
| `GET /stats` | Bu cihazda ölçülen hız (süre tahminleri için) |
| `POST /jobs` | Üretim işi başlatır (stil, sözler, CoT, seed, CFG, ABC, örnekleme, `ode_steps`, `quantization`) |
| `GET /jobs`, `GET /jobs/{id}` | İş listesi; tek işin aşaması, token sayıları, hatası, sonucu |
| `POST /jobs/{id}/cancel` | İptal; yue2'nin `cancelled` geri çağrısıyla aşama içinde durur |
| `GET /jobs/{id}/audio` | MP3 |
| `POST /uploads` | Referans şarkı yükleme; aynı dosya aynı id'yi alır (SHA-256) |
| `POST /analyze` | Referansı ABC notaya (ve isteğe bağlı sözlere) çevirir; sonuç önbelleğe alınır |

Bir üretimin aşamaları: `queued → loading → abc → semantic → synthesize → decode → saving → done`. Ayrıntı:
[ENTEGRASYON.md §4](ENTEGRASYON.md#4-yue2-bir-şarkıyı-nasıl-üretiyor).

---

## 4. Projeyi ayağa kaldırma

### 4.1 İlk 15 dakika: modelsiz deneme (Mac)

Model ve GPU gerekmez. `--mock` sunucusu kısa bir deneme tonu üretir ve örnek bir analiz döndürür; bütün arayüz
akışı (üretim, ilerleme, iptal, geçmiş, cover, sözleri tanı) bununla denenebilir.

```bash
cd ~/Desktop/yueModel/yue-ui

# yue-ui/venv yoksa bir kez (start_ui_mac.sh ilk çalışmada bunu kendisi yapar):
python3.12 -m venv venv && venv/bin/pip install -r requirements-yue2-ui.txt

# Web arayüzü + mock sunucu (7862), tek komut:
./frontend/start.sh --mock                                 # http://127.0.0.1:7870

# ya da Gradio arayüzü, iki terminalde:
./start_model_server.sh --mock --outputs outputs/mock      # :7861
./start_ui_mac.sh                                          # http://127.0.0.1:7860

# Testler (model gerekmez):
./frontend/test.sh
```

### 4.2 Mac'te gerçek model

Bu Mac'te ortamlar ve ağırlıklar kurulu. İki terminalde:

```bash
./start_model_server.sh --preload     # :7861, ağırlıklar ~11 sn'de yüklenir
./frontend/start.sh                   # :7870  (ya da ./start_ui_mac.sh, :7860)
curl -s 127.0.0.1:7861/health         # "model_loaded": true, "device": "mps"
```

- `start_model_server.sh` ağırlıkların varlığını kontrol eder, çevrimdışı değişkenleri ve
  `PYTORCH_ENABLE_MPS_FALLBACK=1`'i ayarlar. Ağırlıklar başka yerdeyse: `YUE2_MODELS=/yol ./start_model_server.sh`.
- **Mac yavaştır:** ~6.5× gerçek zaman (49.5 sn'lik şarkı 5 dk 12 sn). CUDA Graph, FlashAttention, FP8 yok.
- **Uzun şarkı sınırı:** 24 GB M4'te ~6 dakikalık bir şarkı, 43 dakikalık token üretiminden sonra ses sentezinde
  `MPS backend out of memory` ile düştü. Mac'te denemeler için 1–3 dakikalık şarkı (8–20 satır söz) ve
  **⚡ Hızlı sentez** kullanın; uzun şarkıları Colab'da üretin.

**Mac'te sıfırdan kurulum** (tek bir kurulum betiği yok; aşağıdaki komutlar `setup_windows.bat` ve notebook'un 4.
hücresindeki paket listesinden türetildi, bu sırayla sıfırdan denenmedi):

```bash
mkdir -p ~/Desktop/yueModel && cd ~/Desktop/yueModel
git clone https://github.com/batuhansahin1/yueModelDemo.git yue-ui

# 1. Model ortamı (Python 3.12). yue2 kendi sabit sürümlerini kurar; bunlar elle değiştirilmemeli.
python3.12 -m venv yue_ortam
yue_ortam/bin/pip install yue-ui/wheels/yue2_infer-0.1.5-py3-none-any.whl \
    fastapi==0.136.1 uvicorn==0.46.0 torchaudio==2.10.0 librosa "julius>=0.2.3" dora-search einops pyyaml omegaconf
yue_ortam/bin/pip install --no-deps demucs==4.0.1 openunmix

# 2. Ağırlıklar (~12 GB + MERT-v2 2.4 GB). Yalnızca eksik olanı indirir, yarıda kalırsa devam eder.
cd yue-ui
../yue_ortam/bin/python download_models.py ../models --sheetsage

# 3. SheetSage2 ortamı (Python 3.11, torch 2.8)
python3 -m pip install uv
UV_PYTHON_INSTALL_DIR=../.python uv venv --python 3.11 ../sheetsage_ortam
uv pip install --python ../sheetsage_ortam/bin/python torch==2.8.0 torchaudio==2.8.0 \
    -r ../models/SheetSage2/requirements.txt

# 4. Doğrulama: sürümler, cihaz, ağırlıkların SHA-256 özetleri
../yue_ortam/bin/yue2 doctor --model ../models/YuE2-3B --vae ../models/YuE2-Vae --offline --verify-hashes
```

Gradio ortamı (`venv`) `./start_ui_mac.sh` ilk çalıştırıldığında kurulur. Model kurulumunun adım adım anlatımı ve
CUDA/MPS farkları: [yapay-zeka-model-entegrasyonu.md](yapay-zeka-model-entegrasyonu.md).

### 4.3 Windows (NVIDIA)

**Henüz gerçek bir Windows makinesinde denenmedi.** Hedef makine RTX 3050 8 GB, monitör karta bağlı.

1. Kısa, boşluksuz bir klasöre klonla: `mkdir C:\yueModel && cd C:\yueModel && git clone <depo> yue-ui`.
   Mac'teki `models` klasörü `C:\yueModel\models` olarak kopyalanabilir (USB exFAT/NTFS olmalı; FAT32 7.3 GB'lık
   dosyayı almaz). Ortam klasörleri (`yue_ortam`, `sheetsage_ortam`, `venv`, `.python`) **kopyalanmaz**.
2. `setup_windows.bat` (bir kez, internet gerekir; tekrar çalıştırmak güvenli). uv'yi, CUDA'lı torch 2.10'u
   (`TORCH_CUDA`, varsayılan `cu128`), eksik ağırlıkları ve SheetSage2 ortamını kurar. SheetSage2 istenmezse
   `--no-sheetsage`. Windows'ta tek ortam (`yue_ortam`) hem sunucuyu hem iki arayüzü çalıştırır.
3. NVIDIA Denetim Masası → *CUDA - Sysmem Fallback Policy* = *Driver Default* (sürücü 536.40+).
4. Modelsiz deneme: `frontend\start.bat --mock`. Gerçek model, iki pencerede: `start_model_server.bat --preload` ve
   `frontend\start.bat` (ya da `start_ui_windows.bat`).
5. İlk denemede sunucu penceresindeki `--budget auto: ...` satırına (boş VRAM, açılan bayraklar) ve iş süresine bak.

Ayrıntı: [README › Windows](../README.md#windows), [rtx3050-windows-raporu.md](rtx3050-windows-raporu.md).

### 4.4 Windows: başlangıç sihirbazı

4.3'teki adımların kontrol noktalı, ayrıntılı hâli; hedef makine aynı (RTX 3050 8 GB, monitör karta bağlı).

Adımları sırayla izle. Her adımın **✔** satırındaki çıktıyı görmeden sonrakine geçme; görmezsen aynı adımın **✖**
satırına bak. Adım 0–4 bir kez yapılır; sonraki günlerde adım 6 yeterlidir (*Her gün* kutusu). Komutlar **cmd**
(Komut İstemi) içindir. Windows'ta tek ortam (`yue_ortam`) hem model sunucusunu hem iki arayüzü çalıştırır.

**Adım 0 — Ön koşullar**

| Gerekli | Nasıl kontrol edilir |
|---|---|
| NVIDIA ekran kartı, güncel sürücü (paylaşımlı bellek için 536.40+) | `nvidia-smi` → kart adı ve sürücü sürümü |
| Git for Windows | `git --version` |
| ~30 GB boş disk: ortamlar + ~15 GB ağırlık (`models` 12 GB + MERT-v2 2.4 GB). MERT-v2, `C:` altındaki Hugging Face önbelleğine iner | Dosya Gezgini → Bu bilgisayar |
| ~16 GB RAM (`--phase-offload` modelin yarısını RAM'de bekletir) | Görev Yöneticisi → Performans |
| İnternet, yalnızca kurulumda (sonra her şey çevrimdışı çalışır) | |

Python, CUDA araç seti (toolkit) ya da ffmpeg kurmak **gerekmez**: uv Python'u indirir, PyTorch paketleri CUDA'yı
kendisi getirir, MP3 okuma/yazma soundfile ile yapılır (Mac'te ffmpeg olmadan çalışıyor).

✖ `nvidia-smi` tanınmıyorsa sürücü kurulu değildir: NVIDIA'nın sitesinden kur, bilgisayarı yeniden başlat. AMD ya da
Intel kartta model CPU'da çalışır ve kullanılamayacak kadar yavaştır.

**Adım 1 — Klasör ve depo**

```bat
mkdir C:\yueModel
cd /d C:\yueModel
git clone https://github.com/batuhansahin1/yueModelDemo.git yue-ui
cd yue-ui
```

- Kısa, boşluksuz, Türkçe karakter içermeyen bir yol kullan. Ağırlıklar ve ortamlar `yue-ui`'nin yanına
  (`C:\yueModel\`) kurulur.
- Depoyu GitHub'dan ZIP olarak değil, `git clone` ile al. `.gitattributes`, `.bat` dosyalarını CRLF satır sonuyla
  çıkarır; LF satır sonlu `.bat`'larda `cmd.exe` etiketleri (`goto`, `call :etiket`) yanlış okuyabilir.
- Masaüstü ya da Belgeler klasörüne kurma: bunlar çoğu zaman OneDrive ile eşitlenir ve OneDrive ~15 GB ağırlığı ve
  binlerce ortam dosyasını yüklemeye çalışır. `C:\yueModel` OneDrive dışındadır.

✔ `dir` çıktısında `setup_windows.bat` ve `start_model_server.bat` var.
✖ `'git' is not recognized…`: Git for Windows'u kur ve **yeni** bir cmd penceresi aç.

**Adım 2 — (İsteğe bağlı) Ağırlıkları Mac'ten getir**

İnternet yavaşsa Mac'teki `yueModel/models` klasörünü `C:\yueModel\models` olarak kopyala. USB disk exFAT ya da NTFS
olmalı; FAT32 7.3 GB'lık dosyayı almaz. Ortam klasörleri (`yue_ortam`, `sheetsage_ortam`, `venv`, `.python`)
**kopyalanmaz**, Windows'ta çalışmazlar. MERT-v2 Hugging Face önbelleğinde durduğu için kurulumda yine iner. `C:`
darsa kurulumdan önce `setx HF_HOME D:\hf_cache` çalıştırıp yeni bir pencere aç.

✔ `dir C:\yueModel\models\YuE2-3B\model.safetensors` dosyayı gösteriyor.

**Adım 3 — Kurulum (`setup_windows.bat`)**

Betiği çift tıklayarak değil, açık bir cmd penceresinden çalıştır; hata olursa mesaj ekranda kalır.

```bat
cd /d C:\yueModel\yue-ui
setup_windows.bat
```

SheetSage2 istenmezse `setup_windows.bat --no-sheetsage` (Cover'da *Basit analiz* kullanılır). Betik beş adımda
ilerler:

| Satır | Ne yapar | ✔ Görmen gereken |
|---|---|---|
| `=== 1/5 Ekran karti` | Kartı yazar | `NVIDIA GeForce RTX 3050, 8192 MiB, <sürücü>` |
| `=== 2/5 uv` | uv yoksa PowerShell ile kurar | Hata yok |
| `=== 3/5 YuE2 ortami` | `..\yue_ortam`: Python 3.12, CUDA'lı torch 2.10, yue2, Demucs, Gradio | `torch 2.10.0+cu128 \| CUDA True \| yue2 0.1.5`. **`CUDA False` ise devam etme** |
| `=== 4/5 Model agirliklari` | `download_models.py`: eksik ağırlıkları indirir (~15 GB) | Her model için `Var:` ya da `İndiriliyor:` |
| `=== 5/5 SheetSage2 ortami` | `..\sheetsage_ortam`: Python 3.11, torch 2.8 | `SheetSage2 hazir \| torch 2.8.0+cu128 \| CUDA True` |
| Son | | `Kurulum tamam.` |

✖ Olmazsa:
- **Yarıda kaldı** (bağlantı koptu, pencere kapandı): aynı komutu tekrar çalıştır. Kurulu olanlar atlanır, indirme
  kaldığı yerden sürer.
- **`uv kurulamadi`**: PowerShell ya da antivirüs engellemiş olabilir. uv'yi
  [kurulum sayfasından](https://docs.astral.sh/uv/getting-started/installation/) elle kur, yeni pencerede tekrar
  çalıştır.
- **`CUDA False`**: önce sürücüyü güncelle. Düzelmezse ortamı silip eski CUDA sürümüyle kur. Kurulu torch aynı sürüm
  numarasını (`2.10.0`) taşıdığı için yalnızca `TORCH_CUDA`'yı değiştirmek yetmez, torch yeniden inmez:
  `rmdir /s /q ..\yue_ortam` → `set TORCH_CUDA=cu126` → aynı pencerede `setup_windows.bat`. SheetSage2'de de aynı
  sorun varsa `..\sheetsage_ortam` da silinir.
- **`… Mac/Linux'tan kopyalanmis bir ortam`**: `rmdir /s /q ..\yue_ortam` (ya da `..\sheetsage_ortam`), tekrar
  çalıştır.
- **`Uyari: SheetSage2 kurulamadi`**: Text to Song ve Cover yine çalışır; Cover'da *Nota çıkarıcı: Basit analiz* seç.

**Adım 4 — NVIDIA Denetim Masası**

Masaüstüne sağ tık → *NVIDIA Denetim Masası* → *3D ayarlarını yönet* → *CUDA - Sysmem Fallback Policy* =
**Driver Default**. *Prefer No Sysmem Fallback* seçiliyse karta sığmayan uzun şarkılar yavaşlamak yerine bellek
hatası verir. Panel Türkçeyse ayarın adı çevrilmiş olabilir; listede "Sysmem Fallback" ya da "sistem belleği" geçen
CUDA satırını ara. Ayar listede hiç yoksa sürücü 536.40'tan eskidir.

**Adım 5 — Modelsiz deneme**

```bat
frontend\start.bat --mock
```

✔ `Mock model sunucusu: http://127.0.0.1:7862 …` satırı çıkar. Tarayıcıda http://127.0.0.1:7870 → Text to Song →
*Şarkı üret*: birkaç saniyede kısa bir deneme tonu çalar. Bu, ortamın ve arayüzün sağlam olduğunu modeli beklemeden
gösterir. Ctrl+C ile kapat; mock sunucu da kapanır.
✖ `Mock sunucusu baslamadi; port 7862 dolu olabilir`: `set YUE2_MOCK_PORT=7863`, tekrar.

Windows'ta otomatik test betiği yok: `frontend/test.sh` bash betiğidir ve `venv/bin/python`'u arar. Windows'taki
doğrulama bu adımdaki mock denemesidir; testler Mac'te ya da Ubuntu'da çalıştırılır.

**Adım 6 — Gerçek model (iki pencere)**

Pencere 1, model sunucusu:

```bat
cd /d C:\yueModel\yue-ui
start_model_server.bat --preload
```

✔ 8 GB kartta açılışta şu satırlar görünür (sayılar karta ve o anki boş belleğe göre değişir):

```
--budget auto: NVIDIA GeForce RTX 3050, … GiB (… GiB boş) -> --budget … --offload-ar --phase-offload --shared-gpu-memory
Faz bazlı yerleşim: token üretiminde modelin AR yarısı (~4.4 GB), ses sentezinde NAR yarısı (~2.8 GB) …
INFO:     Uvicorn running on http://127.0.0.1:7861 (Press CTRL+C to quit)
```

Ağırlıklar arka planda yüklenir. Hazır olup olmadığını ikinci bir pencerede sor (PowerShell'de `curl` yerine
`curl.exe` yaz):

```bat
curl -s http://127.0.0.1:7861/health
```

`"model_loaded":true` ve `"device":"cuda…"` görünmeli; SheetSage2 kurulduysa `"sheetsage":true`.

Pencere 2, arayüz: `frontend\start.bat` → http://127.0.0.1:7870 (web arayüzü) ya da `start_ui_windows.bat` →
http://127.0.0.1:7860 (Gradio).

✖ Olmazsa:
- **`Model agirliklari bulunamadi`**: `C:\yueModel\models\YuE2-3B\model.safetensors` yok; adım 3'ü tekrar çalıştır.
- **`Python ortami bulunamadi`**: adım 3 tamamlanmamış.
- **`Ekran kartında yalnızca … GiB boş`**: tarayıcı, oyun gibi kartı kullanan uygulamaları kapat. Monitör karta bağlı
  olduğu için masaüstü de VRAM kullanır.
- **`Bu PyTorch FlashAttention olmadan derlenmiş`**: hata değil; token üretimi standart attention'la, biraz daha yavaş
  çalışır ([rtx3050-windows-raporu.md §5](rtx3050-windows-raporu.md#5-flashattention-yedeği)).
- **Arayüz "Model sunucusu bağlı değil" diyor**: pencere 1 açık mı, `/health` yanıt veriyor mu bak. Arayüz sunucuyu
  5 sn'de bir yoklar; sunucu açılınca kendiliğinden bağlanır.

**Adım 7 — İlk gerçek üretim ve ölçüm**

1. **Text to Song**, kısa: 8–12 satır söz, ⚡ Hızlı sentez açık, diğer ayarlar varsayılan.
2. **Cover**, ~1 dakikalık bir bölüm (*Başlangıç* / *Süre*), Nota çıkarıcı *SheetSage2*.
3. Üretim sürerken Görev Yöneticisi → Performans → GPU: *Paylaşılan GPU belleği* artıyorsa karta sığmayan kısım
   sistem RAM'ine taşıyor demektir. Üretim yavaşlar ama hata vermez; CFG'yi 1'de tut, şarkıyı kısalt.
4. Not al: açılıştaki `--budget auto: …` satırı, şarkı süresi ve üretim süresi. Hata çıktıysa sunucu penceresinin
   son 20–30 satırı.

**Uyku ayarı.** Üretim sürerken bilgisayar uykuya geçerse iş yarıda kalır ve sunucu yeniden açılınca `failed`
görünür. *Ayarlar → Sistem → Güç* altında, prize takılıyken uyku süresini *Hiçbir zaman* yap.

**(İsteğe bağlı) Aynı ağdaki başka cihazdan erişim.** Sunucular varsayılan olarak yalnızca bu bilgisayardan açılır.
Telefondan ya da başka bir bilgisayardan denemek için yalnızca web arayüzü ağa açılır:
`frontend\start.bat --host 0.0.0.0 --password PAROLA` (parola özel karakter içeriyorsa önce
`set YUE2_WEB_PASSWORD=PAROLA`). Windows Güvenlik Duvarı soru sorarsa yalnızca *Özel ağlar*'a izin ver. Adres:
`ipconfig` çıktısındaki IPv4 adresi, `http://<IPv4>:7870`. Model sunucusu (7861) ağa açılmaz.

> **Her gün (kurulumdan sonra):** pencere 1'de `start_model_server.bat --preload`, pencere 2'de `frontend\start.bat`,
> tarayıcıda http://127.0.0.1:7870. Kapatmak için iki pencerede Ctrl+C. Kod güncellemesi: `git pull`;
> `setup_windows.bat`'taki paket listesi değiştiyse kurulumu bir kez daha çalıştır (kurulu olanlar atlanır).

Ayrıntı: [README › Windows](../README.md#windows), [rtx3050-windows-raporu.md](rtx3050-windows-raporu.md).

### 4.5 Ubuntu: kurulum ve başlatma

**Hiç denenmedi; kurulum betiği yok.** Kod Linux'u destekler: model sunucusu ortam yollarını (`bin/python`) kendisi
seçer, Mac'teki `.sh` betikleri Linux'ta da çalışır. Aşağıdaki adımlar `setup_windows.bat`'ın Linux karşılığıdır ve
onun paket listesini kullanır. Hedef: Ubuntu 22.04 ya da 24.04 LTS, NVIDIA kart. AMD kartlar (ROCm) desteklenmiyor.

**Gerekenler:** NVIDIA kart ve sürücüsü, ~40 GB boş disk (iki CUDA'lı torch ortamı + ~15 GB ağırlık), ~16 GB RAM,
kurulum sırasında internet ve `sudo` yetkisi. Python, CUDA araç seti ve ffmpeg gerekmez.

> **WSL2 ise** (Ubuntu, Windows makinesinin içinde çalışıyorsa): adım 1'deki sürücü kurulumu **atlanır**. WSL'de
> GPU'yu Windows'taki NVIDIA sürücüsü sağlar, WSL içine Linux NVIDIA sürücüsü kurulmaz; `nvidia-smi` doğrudan
> çalışmalı. Depoyu ve ağırlıkları `/mnt/c` altında değil Linux ev klasöründe (`~/yueModel`) tut: `/mnt/c` üzerinden
> dosya okumak çok yavaştır. Windows'ta inmiş ağırlıklar `cp -r /mnt/c/yueModel/models ~/yueModel/` ile alınabilir.
> Windows tarayıcısı WSL'deki sunucuya `http://127.0.0.1:7870` ile ulaşır. systemd servisleri için `/etc/wsl.conf`'ta
> `[boot]` altında `systemd=true` olmalı. Sunucu WSL'de Linux gibi davranır (`--shared-gpu-memory` kendiliğinden
> açılmaz); WSL'de paylaşımlı GPU belleğine taşma olup olmadığı denenmedi.

**Windows'tan farkları:**

| Konu | Windows | Ubuntu |
|---|---|---|
| Betikler | `.bat` | `.sh`: `start_model_server.sh`, `frontend/start.sh`, `start_ui_mac.sh` (adına rağmen Linux'ta da çalışır) |
| Kurulum | `setup_windows.bat` | Yok; aşağıdaki adımlar elle |
| Ortamlar | Tek ortam (`yue_ortam`) | Mac'teki gibi üç ortam: `yue_ortam` (model), `yue-ui/venv` (Gradio, mock sunucu, testler; `.sh` betikleri bunu arar), `sheetsage_ortam` |
| `--budget auto` | `start_model_server.bat` kendisi verir | **`start_model_server.sh` vermez**, bütçe 24 GiB kalır. 24 GB'tan küçük kartta elle eklenmeli |
| Karta sığmayan bellek | Sürücü sistem RAM'ine taşır (yavaşlar) | Taşma yok: doğrudan `CUDA out of memory` |
| CUDA araç seti (toolkit) | Gerekmez | Gerekmez; yalnızca sürücü. PyTorch paketleri CUDA'yı kendisi getirir |
| FlashAttention | PyTorch paketinde olmayabilir (yedek yol var) | Linux paketlerinde var |
| `--backend vllm` | Yok | Mümkün (`yue2[fast]`), denenmedi. vLLM kendi torch sürümünü isteyebilir; ayrı bir ortamda denenmeli |

**1. Sürücü ve sistem paketleri**

```bash
sudo apt update && sudo apt install -y git curl
ubuntu-drivers devices            # "recommended" yazan sürücü
sudo ubuntu-drivers install       # eski sürümlerde: sudo ubuntu-drivers autoinstall
sudo reboot
nvidia-smi                        # kart adı ve sürücü sürümü görünmeli
```

Secure Boot açıksa kurulum bir MOK parolası ister; yeniden başlatmada mavi ekranda *Enroll MOK* seçilip bu parola
girilmezse sürücü yüklenmez (bkz. 4.6).

**2. uv** (Python'u ve paketleri kurar; Ubuntu'nun Python sürümünden bağımsız olarak 3.12 ve 3.11'i kendisi indirir)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env       # ya da yeni bir terminal
uv --version
```

Ubuntu 23.04 ve sonrasında sistem Python'una `pip install` yapılamaz (PEP 668). 4.2'deki Mac adımı
`python3 -m pip install uv` burada hata verir; her şey uv ile ortamlara kurulur.

**3. Depo**

```bash
mkdir -p ~/yueModel && cd ~/yueModel
git clone https://github.com/batuhansahin1/yueModelDemo.git yue-ui
cd yue-ui
```

Aşağıdaki bütün komutlar `~/yueModel/yue-ui` içinde ve **aynı terminalde** çalıştırılır (`TORCH_INDEX` 7. adımda
tekrar kullanılıyor).

**4. Model ortamı** (`../yue_ortam`; `setup_windows.bat` 3/5 adımının karşılığı)

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu128     # sürücü eskiyse cu126
uv venv --python 3.12 ../yue_ortam
uv pip install --python ../yue_ortam/bin/python torch==2.10.0 torchaudio==2.10.0 --index-url $TORCH_INDEX
uv pip install --python ../yue_ortam/bin/python wheels/yue2_infer-0.1.5-py3-none-any.whl \
    fastapi==0.136.1 uvicorn==0.46.0 librosa "julius>=0.2.3" dora-search einops pyyaml omegaconf
uv pip install --python ../yue_ortam/bin/python --no-deps demucs==4.0.1 openunmix
../yue_ortam/bin/python -c "import torch, yue2, demucs.pretrained, librosa; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| yue2', yue2.__version__)"
```

✔ `torch 2.10.0+cu128 | CUDA True | yue2 0.1.5`. Torch önce CUDA dizininden kurulur; yue2'nin `torch==2.10.0`
şartını karşıladığı için sonraki komut onu değiştirmez.

**5. Arayüz ortamı** (`venv`: Gradio, mock sunucu ve testler)

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python -r requirements-yue2-ui.txt
```

`start_ui_mac.sh`, `venv` yoksa onu sistemdeki `python3.12 -m venv` ile kurmaya çalışır ve Ubuntu'da bu çoğunlukla
başarısız olur. `venv`'i bu adımla önceden kurmak bunu önler.

**6. Ağırlıklar** (~15 GB; yalnızca eksik olan iner, yarıda kalırsa kaldığı yerden sürer)

```bash
../yue_ortam/bin/python download_models.py ../models --sheetsage
```

İnternet yavaşsa Mac'ten kopyalanabilir. Mac'te *Sistem Ayarları → Genel → Paylaşma → Uzaktan Giriş* açılır, sonra
Ubuntu'da:

```bash
rsync -aP KULLANICI@MAC_ADRESI:Desktop/yueModel/models/ ~/yueModel/models/
mkdir -p ~/.cache/huggingface/hub
rsync -aP KULLANICI@MAC_ADRESI:.cache/huggingface/hub/models--m-a-p--MERT-v2-FullSong ~/.cache/huggingface/hub/
../yue_ortam/bin/python download_models.py ../models --sheetsage    # her model için "Var:" yazmalı
```

**7. SheetSage2 ortamı** (isteğe bağlı; kurulmazsa Cover'da *Basit analiz* kullanılır. 6. adımdan sonra çalışır,
çünkü paket listesi `models/SheetSage2/requirements.txt`'tedir)

```bash
uv venv --python 3.11 ../sheetsage_ortam
uv pip install --python ../sheetsage_ortam/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url $TORCH_INDEX
uv pip install --python ../sheetsage_ortam/bin/python -r ../models/SheetSage2/requirements.txt
../sheetsage_ortam/bin/python -c "import torch, transformers; print('SheetSage2 hazır | torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
```

**8. Doğrulama** (sürümler, cihaz, ağırlıkların SHA-256 özetleri)

```bash
../yue_ortam/bin/yue2 doctor --model ../models/YuE2-3B --vae ../models/YuE2-Vae --offline --verify-hashes
```

**9. Başlatma**

Modelsiz deneme (model ve GPU gerekmez):

```bash
./frontend/start.sh --mock        # http://127.0.0.1:7870
./frontend/test.sh server         # isteğe bağlı: web sunucusu + mock sunucu testleri
```

Gerçek model, iki terminalde:

```bash
./start_model_server.sh --preload --budget auto     # --budget auto'yu unutma (yukarıdaki tablo)
./frontend/start.sh                                  # web arayüzü :7870; ya da ./start_ui_mac.sh (Gradio, :7860)
curl -s 127.0.0.1:7861/health                        # "model_loaded":true, "device":"cuda…"
```

8–12 GB kartta açılış satırı `--offload-ar --phase-offload` gösterir ve ardından *"paylaşımlı GPU belleği yalnızca
Windows'ta var"* uyarısı gelir. Bu beklenen bir durumdur; bellek hatası alınırsa 4.6'daki OOM satırlarına bak.

**Başka bir bilgisayardan erişim.** Sunucular yalnızca `127.0.0.1`'i dinler. Ubuntu makinesi ekransız bir sunucuysa
en güvenli yol SSH tünelidir; kendi bilgisayarında:

```bash
ssh -L 7870:127.0.0.1:7870 KULLANICI@UBUNTU_ADRESI      # sonra tarayıcıda http://127.0.0.1:7870
```

Ağa açmak gerekirse yalnızca web arayüzü açılır: `./frontend/start.sh --host 0.0.0.0 --password PAROLA` ve
`sudo ufw allow 7870/tcp`. İnternete açılacaksa önüne TLS'li bir ters vekil şarttır (bkz. bölüm 9, "Herkese açık
site"). Model sunucusu (7861) hiçbir zaman dışarı açılmaz.

**Kalıcı çalıştırma (systemd).** Terminalde başlatılan sunucular SSH oturumu kapanınca kapanır. Kısa denemelerde
`tmux` yeter; sürekli çalışacak bir sunucuda iki servis tanımlanır (`KULLANICI` yerine kendi kullanıcı adın):

```ini
# /etc/systemd/system/yue2-model.service
[Unit]
Description=YuE2 model sunucusu (:7861)
After=network.target

[Service]
User=KULLANICI
WorkingDirectory=/home/KULLANICI/yueModel/yue-ui
Environment=PYTHONUTF8=1 PYTHONUNBUFFERED=1
ExecStart=/home/KULLANICI/yueModel/yue-ui/start_model_server.sh --preload --budget auto --keep-loaded
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/yue2-web.service
[Unit]
Description=YuE2 web arayüzü (:7870)
After=yue2-model.service

[Service]
User=KULLANICI
WorkingDirectory=/home/KULLANICI/yueModel/yue-ui
Environment=PYTHONUTF8=1 PYTHONUNBUFFERED=1
ExecStart=/home/KULLANICI/yueModel/yue-ui/frontend/start.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now yue2-model yue2-web
systemctl status yue2-model
journalctl -u yue2-model -f        # sunucu logu (Ctrl+C yalnızca log izlemeyi bırakır)
```

- `--keep-loaded`: ağırlıklar işler arasında GPU'da kalır (Colab'daki gibi); ilk iş beklemez. Kart başka işlerle
  paylaşılacaksa kaldırılır.
- `HF_HOME` değiştirildiyse model servisine `Environment=HF_HOME=/yol` eklenir; yoksa servis MERT-v2'yi bulamaz.
- Servisler `User=` ile kurulumu yapan kullanıcı olarak çalışmalı: ortamlar, ağırlıklar ve Hugging Face önbelleği o
  kullanıcının ev klasöründe.
- Masaüstü Ubuntu'da *Ayarlar → Güç* altında otomatik askıya alma kapatılmalı; askıya alınan makinede süren iş
  yarıda kalır ve `failed` olur.

> **Her gün (kurulumdan sonra):** servis yoksa iki terminalde `./start_model_server.sh --preload --budget auto` ve
> `./frontend/start.sh`; kapatmak için Ctrl+C. Servis varsa makine açılınca kendiliğinden başlar; elle
> `sudo systemctl stop|start|restart yue2-model yue2-web`. Kod güncellemesi: `git pull`; paket listesi değiştiyse 4, 5
> ve 7. adımlardaki `uv pip install` komutlarını tekrar çalıştır (kurulu olanlar atlanır), sonra
> `sudo systemctl restart yue2-model yue2-web`.

### 4.6 Ubuntu: olası sorunlar ve çözümleri

Ubuntu'da hiç çalıştırılmadığı için bu tablo koddan ve betiklerden çıkarıldı; ilk kurulumda karşılaşılanlar buraya
eklenmeli.

**Kurulum**

| Belirti | Neden | Çözüm |
|---|---|---|
| `nvidia-smi: command not found` ya da `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | Sürücü kurulu değil ya da Secure Boot imzasız çekirdek modülünü engelledi | `sudo ubuntu-drivers install`, yeniden başlat. `mokutil --sb-state` "enabled" diyorsa yeniden başlatmada *Enroll MOK* adımını tamamla ya da BIOS'ta Secure Boot'u kapat |
| `error: externally-managed-environment` | Ubuntu 23.04+ sistem Python'una pip ile kurulumu engelliyor (PEP 668) | Sistem pip'ini kullanma: uv'yi `curl` ile kur (4.5 adım 2), paketleri `uv pip install --python <ortam>/bin/python` ile ortama kur |
| `uv: command not found` (kurulumdan hemen sonra) | `~/.local/bin` PATH'te değil | `source $HOME/.local/bin/env` ya da yeni terminal |
| `./start_ui_mac.sh` → `python3.12: command not found` ya da `ensurepip is not available` | `venv` yoksa betik sistemdeki `python3.12 -m venv`'i dener. 22.04'te Python 3.12 yok; 24.04'te `python3.12-venv` paketi kurulu değil | `rm -rf venv`, sonra 4.5 adım 5 (uv ile `venv`). Yalnızca 24.04'te alternatif: `sudo apt install python3.12-venv` |
| SheetSage2 kurulumunda `numpy==1.24.3` derlenmeye çalışılıyor, `Failed to build numpy` | Ortam Python 3.11 değil; numpy 1.24.3'ün 3.12 için hazır paketi yok | `rm -rf ../sheetsage_ortam`, `uv venv --python 3.11 ../sheetsage_ortam` ile yeniden (adım 7) |
| `/bin/bash^M: bad interpreter` | `.sh` dosyası CRLF satır sonlu: Windows'tan kopyalandı ya da ZIP ile geldi | `sed -i 's/\r$//' *.sh frontend/*.sh colab/*.sh`; ya da depoyu `git clone` ile al (`.gitattributes` `.sh`'leri LF tutar) |
| `./start_model_server.sh: Permission denied` | Çalıştırma izni kopyalarken kayboldu (USB, ZIP) | `chmod +x *.sh frontend/*.sh colab/*.sh` |
| `../yue_ortam/bin/python: No such file or directory` ya da `cannot execute binary file` | Ortam Mac'ten ya da Windows'tan kopyalandı. Ortamlar taşınamaz; Mac'teki `bin/python` macOS'un Python'unu gösterir | Ortam klasörlerini sil, 4.5 adım 4, 5, 7 ile yeniden kur. Makineler arasında yalnızca `models/` (ve MERT-v2 önbelleği) kopyalanır |
| `./start_model_server.sh` → `../yue_ortam/bin/pip: No such file or directory` | Betik `fastapi` eksikse onu `pip` ile kurmaya çalışır; uv'nin kurduğu ortamlarda `pip` yoktur | `uv pip install --python ../yue_ortam/bin/python fastapi==0.136.1 uvicorn==0.46.0` (4.5 adım 4 bunu zaten kurar) |
| `No space left on device` | İki CUDA'lı torch ortamı (her biri birkaç GB), uv önbelleği, ~15 GB ağırlık | `df -h`; `uv cache clean`. Hugging Face önbelleğini büyük diske taşımak için `export HF_HOME=/disk/hf` (servise de). ~40 GB boş yer ayır |

**GPU ve model sunucusu**

| Belirti | Neden | Çözüm |
|---|---|---|
| `/health`'te `"device":"cpu"`, açılışta `--budget auto: NVIDIA GPU bulunamadı` | torch CUDA'yı görmüyor: sürücü yok ya da eski, veya torch'un CPU sürümü kurulu | `../yue_ortam/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. Sürüm `+cpu` ise ortamı silip adım 4'ü `--index-url` ile tekrarla. `The NVIDIA driver on your system is too old` uyarısı varsa sürücüyü güncelle ya da ortamı silip `TORCH_INDEX=https://download.pytorch.org/whl/cu126` ile kur (kurulu `2.10.0` yeniden inmez, silmek şart) |
| İlk işte `torch.OutOfMemoryError: CUDA out of memory` (8–16 GB kart) | `start_model_server.sh` `--budget auto` vermez; bütçe 24 GiB kalır, `--offload-ar`/`--phase-offload` açılmaz | `--budget auto` ile başlat. Açılış satırında `--offload-ar` (12 GB altında ayrıca `--phase-offload`) görünmeli |
| `--budget auto` ile de uzun şarkıda `CUDA out of memory` | Linux'ta paylaşımlı GPU belleği yok. CFG > 1 ya da Nota planı *Kapalı* KV önbelleğini iki katına çıkarır | CFG 1, Nota planı açık, şarkıyı kısalt (Cover'da *Süre*). Masaüstü aynı kartı kullanıyorsa `nvidia-smi`'de `Xorg`/`gnome-shell` belleğine bak; makineyi ekransız çalıştırmak (`sudo systemctl set-default multi-user.target`) birkaç yüz MB ile ~1 GB kazandırır |
| `Model ağırlıkları bulunamadı`, klasör var gibi görünüyor | Linux büyük/küçük harf ayırır: `YuE2-Vae` ile `YuE2-VAE` farklıdır (Mac'te fark etmez) | Klasör adları bölüm 2'deki tablodakiyle birebir aynı olmalı |
| Cover'da `SheetSage2 kurulu değil, eksik: …/sheetsage_ortam/bin/python` | Ortam kurulmadı ya da başka yerde | 4.5 adım 7; ya da `--sheetsage-python /yol/bin/python` |
| SheetSage2 analizinde MERT-v2 bulunamadı (`LocalEntryNotFoundError`, `couldn't connect to huggingface.co`) | Sunucu çevrimdışı (`HF_HUB_OFFLINE=1`); MERT-v2 inmemiş ya da servis başka kullanıcı/başka `HF_HOME` ile çalışıyor | `download_models.py ../models --sheetsage`; servis kurulumu yapan kullanıcıyla ve aynı `HF_HOME` ile çalışmalı |
| MP3 hatası (`Format not recognised`, `Error opening … .mp3`) | soundfile eski bir sistem libsndfile'ını kullanıyor; MP3 desteği libsndfile 1.1.0'da geldi | `../yue_ortam/bin/python -c "import soundfile; print(soundfile.__libsndfile_version__)"` en az 1.1.0 olmalı. uv'nin kurduğu `soundfile==0.13.1` kendi libsndfile'ını (1.2.x) getirir; apt'nin `python3-soundfile` paketi ortama karışmamalı |
| `[Errno 98] Address already in use` | Port dolu: önceki sunucu ya da arka planda kalan mock sunucu | `ss -ltnp \| grep 786` ile süreci bul, `kill <pid>`. Mock için `YUE2_MOCK_PORT=7863 ./frontend/start.sh --mock` |

**Erişim ve testler**

| Belirti | Neden | Çözüm |
|---|---|---|
| Başka bilgisayardan açılmıyor | Sunucular yalnızca `127.0.0.1`'i dinler (tasarım gereği) | SSH tüneli (4.5). Ağa açmak: web arayüzü `--host 0.0.0.0 --password …` + `sudo ufw allow 7870/tcp`. Gradio `./start_ui_mac.sh --server_name 0.0.0.0` ile açılır ama parolası yoktur. 7861 açılmaz |
| SSH kapanınca sunucu da kapanıyor | Süreç terminale bağlı | `tmux` ya da systemd servisleri (4.5) |
| `./frontend/test.sh` Node testlerini atlıyor ya da e2e hata veriyor | Ubuntu deposundaki Node eski: 22.04'te 12, 24.04'te 18. Mantık testleri Node 18+, e2e Node 22+ ister | Node 22'yi nvm ya da NodeSource ile kur |
| `Chrome bulunamadı: tarayıcı testleri atlandı`, ya da e2e Chromium'u açıp takılıyor | Test `/usr/bin/google-chrome`, `/usr/bin/chromium`, `/usr/bin/chromium-browser` arar. Ubuntu'nun Chromium'u snap paketidir; snap'in kendi `/tmp`'si olduğu için testin geçici profil klasörünü göremeyebilir | Google Chrome'un `.deb` paketini kur ya da `CHROME=/yol/chrome ./frontend/test.sh e2e` |

### 4.7 Model sunucusu seçenekleri

| Seçenek | Varsayılan | Açıklama |
|---|---|---|
| `--mock` | kapalı | Model yok; deneme tonu üretir. Yalnızca fastapi + uvicorn ister (`venv` yeterli) |
| `--preload` | kapalı | Modeli başlangıçta yükler (yoksa ilk işte yüklenir) |
| `--model`, `--vae` | HF repo adı | Betikler `../models/YuE2-3B` ve `../models/YuE2-Vae` verir |
| `--offline` | kapalı | Yalnızca indirilmiş ağırlıklar; betikler hep açar |
| `--device` | `auto` | `auto` (cuda > mps > cpu), `cuda`, `mps`, `cpu` |
| `--budget` | `24` | Bellek bütçesi (GiB) ya da `auto`. ≤12 ise VAE 512'lik parçalarla çözülür |
| `--offload-ar` | kapalı | Sentez sırasında AR yarısını boşaltır (16 GB kartlar) |
| `--phase-offload` | kapalı | Her aşamada modelin yalnızca kullanılan yarısı GPU'da, diğeri RAM'de (8–12 GB kartlar). Kalite değişmez |
| `--shared-gpu-memory` | kapalı | yue2'nin "VRAM − 2 GiB" tavanını kaldırır; Windows'ta taşan kısım sistem RAM'ine gider |
| `--keep-loaded` | kapalı | Ağırlıklar işler arasında bellekte kalır. Kapalıyken kuyruk boşalınca boşaltılır (yeniden yükleme ~11 sn). Colab'da açık |
| `--backend` | `torch` | `torch` (CUDA'da CUDA Graph), `torch-eager`, `vllm` (yalnızca NVIDIA, `[fast]` ekstrası, kurulu değil) |
| `--quantization` | `none` | `fp8` yalnızca compute 8.9+ (L4, RTX 40xx). Kalite etkisi denenmedi |
| `--analysis-device` | `auto` | Demucs: CUDA varsa GPU, yoksa CPU (Mac'te CPU daha güvenilir çıktı) |
| `--asr-device` | `auto` | Whisper cihazı |
| `--whisper`, `--whisper-tr` | `../models/whisper-*` | `--whisper-tr ''` Türkçe modeli kapatır (Türkçe de turbo ile yazılır) |
| `--sheetsage`, `--sheetsage-python` | `../models/SheetSage2`, `../sheetsage_ortam/...` | SheetSage2 klasörü ve ortamının Python'u |
| `--outputs`, `--host`, `--port` | `yue-ui/outputs`, `127.0.0.1`, `7861` | |

**`--budget auto`** ([resolve_budget](../server/yue2_server.py#L544)): yalnızca `start_model_server.bat` kullanır;
Ubuntu'da `start_model_server.sh`'ye elle eklenir (4.5).
Bütçe = kartın belleği; 20 GiB altında `--offload-ar`; 13 GiB altında `--phase-offload` (Windows'ta ayrıca
`--shared-gpu-memory`). Mac betiği bütçeyi vermez (24); Colab notebook'u kartın belleğini ve 20 GiB altında
`--offload-ar`'ı kendisi verir.

**Ortam değişkenleri:**

| Değişken | Nerede | Etkisi |
|---|---|---|
| `YUE2_ENV`, `YUE2_MODELS` | başlatma betikleri | Model ortamının ve ağırlıkların yeri (varsayılan `../yue_ortam`, `../models`) |
| `SHEETSAGE_ENV`, `TORCH_CUDA` | `setup_windows.bat` | SheetSage2 ortamı, PyTorch CUDA sürümü (`cu128`, `cu126`) |
| `HF_HOME` | her yer | Hugging Face önbelleği (MERT-v2 burada); C: dar ise taşınır |
| `YUE2_SERVER` | iki arayüz | Model sunucusunun adresi (varsayılan `http://127.0.0.1:7861`) |
| `YUE2_WEB_HOST`, `YUE2_WEB_PORT`, `YUE2_WEB_PASSWORD`, `YUE2_MOCK_PORT` | web arayüzü | `--host`, `--port`, `--password`, mock portu (7862) |
| `GRADIO_ROOT_PATH` | Colab | Gradio'nun ses/indirme linklerini Colab proxy adresiyle kurması için |
| `PYTHONUTF8=1` | Windows betikleri | Türkçe karakterlerin dosya ve loglara bozulmadan yazılması |

---

## 5. Modlar

### 5.1 Hangi modda hangi model çalışır

| Mod | YuE2-3B + VAE | SheetSage2 | Demucs | Whisper turbo | Whisper TR |
|---|---|---|---|---|---|
| **Text to Song** | ✔ (ABC'yi model kendisi yazar) | – | – | – | – |
| **Cover · SheetSage2** (varsayılan) | ✔ (nota planı aşaması atlanır) | ✔ | – | – | – |
| **Cover · SheetSage2 + Sözleri tanı** | ✔ | ✔ | ✔ (yalnızca vokal) | ✔ (dil algılama; Türkçe değilse yazar) | ✔ (dil `tr` ise) |
| **Cover · Basit analiz** | ✔ | – | ✔ | – | – |
| **Cover · Basit analiz + Sözleri tanı** | ✔ | – | ✔ | ✔ | ✔ (dil `tr` ise) |
| **Mock** | Model yok, deneme tonu | – | – | – | – |

Yakın/Serbest seçimi bu tabloyu değiştirmez; yalnızca YuE2'ye giden notayı değiştirir (5.3). Modellerin iç işleyişi
ve aralarındaki veri akışı: [model-calisma-prensibi.md](model-calisma-prensibi.md).

### 5.2 Text to Song

- **Girdiler:** Şarkı açıklaması (≤3000 karakter), sözler (≤5000 karakter, `[verse]`/`[chorus]`/`[bridge]`),
  🎙 Vokal (Varsayılan/Kadın/Erkek), 🎚 Tür (hazır türler + serbest *Özel tür*), CoT, seed, ⚡ Hızlı sentez.
- YuE2 tek bir `[Tags]` metni alır. Arayüz bunu `türler, özel tür, vokal etiketi, açıklama` sırasıyla birleştirir
  ([t2s_style](../source/yue2_ui.py#L430)). `[Verse]` gibi etiketler küçük harfe çevrilir.
- Modele giden metnin tamamı sonuçtaki **Modele giden girdi** alanında görünür. Form girdileri işle birlikte
  (`request.prompt`) saklanır; Geçmiş'ten forma geri yüklenebilir.
- Melodiyi, akorları ve tempoyu model kendisi besteler. MP3 ya da nota çıkarma yoktur.

### 5.3 Cover

Cover'da iki ayrı seçim vardır:

**Nota çıkarıcı** (notayı kim yazar):

| Seçenek | Nasıl çalışır | Ne zaman |
|---|---|---|
| **SheetSage2** (varsayılan) | YuE2 ekibinin modeli. Karışık sesi doğrudan dinler; melodi, akor, vuruş, ton, bölümler ve enstrüman melodisini (`V: Ins`) YuE2'nin ABC biçiminde yazar | Vokalli kayıtlar |
| **Basit analiz** | `cover.py`: Demucs ile kanallara ayırır; librosa ile tempo, ton (Krumhansl), ölçü başına akor; pYIN ile vokal melodisi. Tempo bazen 4/3 ya da 2 kat yanılır; bu yüzden elle BPM girilebilir | Vokalsiz/enstrümantal kayıtlar; SheetSage2 kurulu değilse |

**Referans melodisi** (YuE2'ye ne gider) — [COVER_MODES](../source/yue2_ui.py#L33),
[strip_chords](../source/yue2_ui.py#L419):

| Mod | YuE2'ye giden nota | CoT | Sonuç |
|---|---|---|---|
| **Yakın** | Akorlu nota, olduğu gibi | `full` | Orijinal armoni korunur |
| **Serbest** | Akor sembolleri silinmiş nota (`V: Vocal` ve `V: Ins` melodileri kalır) | `melody` | Eşlik akorlarını model seçer. YuE2 ekibinin stil değiştirirken önerdiği mod |

- **Varsayılanlar farklı:** Gradio'da *Yakın*, web arayüzünde *Serbest*. Proje sahibinin 12 Ekim dinleme
  denemelerindeki tercihi **Serbest**: vokalin okunuşu seçilen türe dönüyor ama eşlik orijinale benzer kalıyor
  (büyük ihtimalle notada kalan `V: Ins` satırları ve orijinal tempo yüzünden). *Yakın*'da tür hissi zayıf kaldı;
  notasız üretimde (MP3 kaldırılarak) tür iyi ama melodi tamamen yeni, yani cover değil.
- Mod, analizden sonra da değiştirilebilir; nota elle düzenlenmediyse yeniden analiz yapılmadan dönüştürülür.
- "SheetSage2 + Serbest", YuE2'nin resmi cover tarifiyle aynıdır. Demucs ve Whisper projenin kendi eklemeleri;
  resmi tarifte elle yapılan söz yazımını otomatikleştirir.

**✨ Sözleri tanı:** YuE2, sözlerdeki n'inci `[bölüm]` bloğunu notadaki n'inci *söylenen* bölümde söyler; zaman
girdisi yoktur. Sözleri tanı, Demucs'un ayırdığı vokali Whisper'la kelime zamanlarıyla yazıya döker ve her satırı
zamanına göre doğru bölüme yerleştirir ([place_in_sections](../server/lyrics.py#L307)). Söz bulunmayan söylenen bölüm
`instrumental`, söz bulunan enstrümantal bölüm `verse` olarak yeniden adlandırılır. Zamanlar YuE2'ye gönderilmez.
Türkçe şarkılarda Türkçe Whisper kullanılır (test kaydında kelime hatası %23.7 → %13.6; Mac'te ~4 kat yavaş).

**Doğrudan "Şarkı üret":** MP3 yüklüyken nota kutusu boşsa "Şarkı üret" önce analizi kendisi yapar
([generate](../source/yue2_ui.py#L755)); bu durumda yalnızca nota çıkarıcı çalışır, Demucs/Whisper çalışmaz. Sözler
zorunludur ve blok sırası notadaki söylenen bölümlerle tutmalıdır; tutmazsa uyarı çıkar. Önerilen akış: *Melodiyi
çıkar (önizle)* → zaman çizelgesinde bölümleri gör → sözleri bu sırayla yaz → *Şarkı üret* (analiz önbellekten gelir).

**Diğer Cover ayrıntıları:**
- *Başlangıç/Süre* ile şarkının bir bölümü seçilebilir (Süre 0 = tümü).
- Şarkı token sınırı referansın süresine göre otomatik büyütülür: `min(20000, max(mevcut, saniye × 25 × 1.3))`.
- Model referans süresine tam uymaz; çıktı referansın 1–1.6 katı uzunlukta olabilir.

### 5.4 CoT (nota planı) modları

| CoT | Ne yapar | Not |
|---|---|---|
| `full` | Önce akorlu ABC nota planı, sonra şarkı | Text to Song varsayılanı; Cover · Yakın |
| `melody` | Akorsuz ABC planı, sonra şarkı | Cover · Serbest |
| `off` | Nota planı yok, doğrudan ses token'ları | Harici ABC ile kullanılamaz. CFG ikinci dalı açık olduğu için KV önbelleği iki katına çıkar |

Dışarıdan ABC verildiğinde (Cover) nota planı aşaması atlanır, model verilen notayı takip eder.

### 5.5 Model ayarları (iş başına)

| Ayar | Varsayılan | Etkisi |
|---|---|---|
| Seed | rastgele | Aynı seed + aynı ayarlar = aynı şarkı (yue2 TF32'yi kapatıp cuDNN'i deterministik çalıştırır) |
| CFG scale | model varsayılanı | >1: stil etiketlerine ve sözlere daha sıkı uyum (notaya değil); şarkı aşaması ~2× yavaş, KV önbelleği 2× |
| Ses sentezi adımı (ODE) | 32 | 4–64. ⚡ Hızlı sentez = 16 (M4'te sentez 3.3 → 2.0 sn/ses saniyesi). Kalite farkı dinlenerek karşılaştırılmadı |
| FP8 (deneysel) | kapalı | Yalnızca compute 8.9+; CUDA Graph kapandığı için yavaş olabilir |
| ABC örneklemesi | temp 0.7, top-p 0.9, top-k 30, ceza 1.005/100, token 32–4096 | Nota planı aşaması |
| Şarkı örneklemesi | temp 1.0, top-p 0.95, top-k 100, ceza 1.2/50, token 200–9000 | Ses token'ları aşaması (~25 token = 1 sn ses; 9000 ≈ 6 dk) |

Belleği ve arka ucu etkileyen ayarlar iş başına değil, sunucu başlatılırken verilir (4.7) ve müziği değiştirmez.

### 5.6 Sunucu bellek modları (donanıma göre)

| Donanım | Önerilen | Notlar |
|---|---|---|
| Mac M4 24 GB (MPS) | varsayılan (`./start_model_server.sh`) | Birleşik bellek; uzun şarkıda swap ve OOM. İşten sonra bellek boşaltılır |
| Colab L4 24 GB | `--budget <kartın GiB'i> --keep-loaded` (notebook verir) | Analizden önce GPU'da 8 GiB'tan az boş yer varsa YuE2 boşaltılır ([make_room](../server/yue2_server.py#L595)) |
| 16 GB (T4 vb.) | `--budget 15 --offload-ar` | T4 BF16'yı donanımda hızlandırmaz, yavaş |
| 8–12 GB (RTX 3050) | `--budget auto` → `--offload-ar --phase-offload --shared-gpu-memory` | Tepe ~6 GiB. Linux'ta paylaşımlı bellek yok: sığmayan uzun şarkı + CFG OOM verir |

`--phase-offload`, yue2 paketine dokunmadan sunucuda uygulanır (`phase_groups`, `place`, `phase_placement`).
Mac'te oyuncak modelle çıktının bit bit aynı olduğu doğrulandı; gerçek kartta ölçülmedi. Ayrıca CUDA'da PyTorch
FlashAttention'sız derlenmişse token üretimi standart SDPA'ya geçer (`without_flash_attention`).

### 5.7 Geri alınan mod: "Sözlerden"

9 Ekim'de Cover'a üçüncü bir mod eklendi: referanstan yalnızca sözler alınır, modele nota gönderilmez. 12 Ekim'de
proje sahibinin isteğiyle geri alındı (melodi tamamen yeni olduğu için cover sayılmadı). **Git'te yoktur;** tek
kopyası Mac'teki `yueModel/yedek-sozlerden-9ekim/` klasörüdür:

| Dosya | İçerik |
|---|---|
| `yue2_ui.py` | Sözlerden modlu arayüz (9 Ekim) |
| `README.md` | O sürümün README'si |

Geri getirilecekse yedekteki `yue2_ui.py` doğrudan kopyalanmamalı: 13 Ekim'deki küçük değişiklikler (Windows
başlatma mesajı) onda yok; farklar elle birleştirilmeli ve web arayüzü de güncellenmeli.

---

## 6. Colab: paketleme ve notebook

### 6.1 Genel akış

```
Mac: ./colab/make_bundle.sh ──► colab/yue2-colab.zip (~120 KB, kod + wheel)
                                      │  Drive'ın ana klasörüne (MyDrive/yue2-colab.zip) ya da hücre 3'te yükle
                                      ▼
Colab (L4): YuE2_Colab.ipynb ──► ortamlar + ağırlıklar (ilk oturumda iner, Drive'daki yue2_cache'e yazılır)
                              ──► model sunucusu :7861 + Gradio :7860 ──► Colab proxy linki
```

Ön koşullar: Google AI Pro (≈200 compute birimi/ay), çalışma zamanı türü **L4 GPU** (yoksa A100; T4 yavaş),
Drive'da ~25 GB boş yer. Kullanıcı adımları ve sorun giderme: [COLAB_YOL_HARITASI.md](../colab/COLAB_YOL_HARITASI.md).

### 6.2 Paket (`make_bundle.sh`)

```bash
cd ~/Desktop/yueModel/yue-ui && ./colab/make_bundle.sh     # → colab/yue2-colab.zip
```

Betik `yueModel/` kökünden çalışır ve zip'e şunları koyar: kökte `yue2_infer-0.1.5-py3-none-any.whl` (notebook onu
`/content`'ten kurar), `yue-ui/` altında `server/*.py` (5 dosya), `source/yue2_ui.py`, `top_200_tags.json`,
`examples/example.json`, `requirements-yue2-ui.txt`, `README.md`. Zip `/content`'e açılır, yani kod
`/content/yue-ui/...` yolunda durur.

**Pakete girmeyenler:** `frontend/`, `docs/`, başlatma betikleri, `download_models.py`, `LICENSE`/`NOTICE`. Ağırlıklar
da yoktur, Colab'da iner. Zip `.gitignore`'da; GitHub'da yoktur, her geliştirici kendisi üretir.

**Ne zaman yeniden üretilir:** Listelenen dosyalardan biri değiştiğinde. Sonra Drive'daki eski
`MyDrive/yue2-colab.zip` yenisiyle **değiştirilmeli**: notebook Drive'da zip bulursa sormadan onu kullanır. Aynı
oturumda yeni zip denenecekse önce *Oturumu yeniden başlat* ya da `/content/yue2-colab.zip`'i sil.

> Not: 13 Ekim'de sunucuya eklenen 8 GB kart ve Windows desteği (`--phase-offload`, FlashAttention yedeği) Colab'da
> henüz çalışmadı. L4'te bu yolların devreye girmemesi beklenir; güncel koddan üretilen ilk paket bir deneme
> oturumuyla doğrulanmalı.

### 6.3 Notebook yapısı (`colab/YuE2_Colab.ipynb`)

Notebook elle yazılmıştır; onu üreten bir betik yoktur. Metadata'daki `"accelerator": "GPU", "gpuType": "L4"`
açılışta L4'ü önerir. Hücreler (*Çalışma zamanı → Tümünü çalıştır* ile sırayla):

| Hücre | Ne yapar | Tanımladığı/kullandığı |
|---|---|---|
| 1. GPU kontrolü | `nvidia-smi`; GPU yoksa durur. Compute < 8.0 ise (T4) uyarır | `GPU_MEM_GIB` |
| 2. Drive | Drive'ı bağlar | `DRIVE_CACHE = True`, `USE_DRIVE = False`, `OUTPUTS`, `YUE2_CACHE` |
| 2 (devam) | `%%writefile /content/cache.sh`: Drive önbellek fonksiyonları (6.4) | `cache_restore`, `cache_save`, `cache_prune` |
| 3. Paketi yükle | `/content/yue2-colab.zip` yoksa `MyDrive/yue2-colab.zip`'i kopyalar, o da yoksa dosya sorar; açar | `BUNDLE`, `DRIVE_BUNDLE` |
| 4. Paketleri kur | uv ile `/content/yue2_ortam` (Python 3.12): wheel + `REQS`, `--no-deps` ile `NODEPS` (demucs, openunmix). Colab'ın kendi paketleriyle çakışmasın diye ayrı ortam. Son satır **Whisper pipeline OK** | `REQS`, `NODEPS`, `KEY` |
| 5. Ağırlıklar | YuE2-3B + VAE (`yue2.storage.MODEL_FILES` desenleriyle), iki Whisper, Demucs → `/content/models` | önbellek parçası `models` |
| 5b. SheetSage2 | Model + MERT-v2 (config'teki sabit revizyon) ve `/content/sheetsage_ortam` (Python 3.11, torch 2.8 cu126). Son satır **SheetSage2 hazır**. Hata verirse Cover'da *Basit analiz* seçilir | önbellek parçaları `sheetsage`, `sheetsage-env-<md5>` |
| 6. Model sunucusu | `yue2_server.py --offline --keep-loaded --budget <VRAM> [--offload-ar]`, SheetSage2 yolları; `HF_HUB_OFFLINE=1`; `/health` gelene kadar bekler | `YUE2_PY`, `server` |
| 7. Arayüz | Colab proxy adresini alır (`proxyPort(7860)`), Gradio'yu `GRADIO_ROOT_PATH` ile başlatır, iframe'de de açar | `UI_URL`, `ui` |
| 7b. (Yedek) | Yorum satırında: Gradio `--share` herkese açık link (72 saat) | |
| 8. Loglar | `/content/logs/model_server.log` ve `ui.log`'un sonu | |
| 8b. (Yedek) | Son üretilen `audio.mp3`'ü doğrudan indirir | |
| 9. Bitir | `KAPAT = False # @param`. İşaretlenince sunucuları kapatır, `drive.flush_and_unmount()` ile Drive yüklemesini bekler, `runtime.unassign()` ile GPU'yu bırakır | |

> **Unutulmaması gereken:** 9. hücre KAPAT işaretli değilse hiçbir şeyi kapatmaz; GPU bağlı kaldıkça üretim olmasa da
> compute birimi harcanır. İlk oturumda KAPAT, ~25 GB'lık önbelleğin Drive'a yüklenmesini bekler; yarıda kesilirse
> sonraki oturumda yeniden indirilir.

### 6.4 Drive önbelleği

Colab her oturumda boş makineyle başlar. `cache.sh` her parçayı Drive'da **tek, sıkıştırmasız bir tar** olarak
saklar (Drive çok sayıda küçük dosyada yavaş):

- **Bütünlük:** Arşivin son üyesi bir işaret dosyasıdır (`/content/.cache_ok/<ad>`). Drive yüklemesi yarıda kaldıysa
  işaret eksik olur; `cache_restore` arşivi siler ve hücre yeniden indirir/kurar.
- **Sürüm anahtarı:** Ortam arşivlerinin adı içeriklerinin özetini taşır: `yue2-env-<md5(wheel + REQS + NODEPS)>`,
  `sheetsage-env-<md5(requirements.txt + TORCH)>`. Paket listesi ya da wheel değişirse yeni anahtar oluşur, ortam
  yeniden kurulur ve `cache_prune` eskisini siler.
- **Ağırlıkların anahtarı yok:** `models.tar` ve `sheetsage.tar` sabit adlıdır. Bir model güncellenirse ya da
  indirilen model listesi değişirse Drive'daki `yue2_cache/models.tar` elle silinmelidir.
- **Tam sıfırlama:** Drive'daki `yue2_cache` klasörünü silmek.

### 6.5 Notebook'u düzenleme ve yeni notebook oluşturma

**Düzenleme yolu:** Notebook'u Colab'da açıp düzenle → *Dosya → İndir → .ipynb indir* → `colab/YuE2_Colab.ipynb`'nin
üzerine yaz. Ya da yerelde VS Code/Jupyter ile düzenle. Depoya koymadan önce hücre çıktılarını temizle (depodaki
notebook'ta çıktı yok). Hücre başlıkları ve numaraları `COLAB_YOL_HARITASI.md`'deki tabloyla aynı kalmalı.

**Senkron tutulması gereken yerler:** Notebook, yerel betiklerin Colab karşılığıdır. Birini değiştiren diğerini de
değiştirmeli:

| Değişiklik | Notebook'ta | Eşleri |
|---|---|---|
| `yue_ortam` paket listesi | Hücre 4: `REQS`, `NODEPS` | `setup_windows.bat` (3/5 adım), Mac kurulumu (4.2), Ubuntu kurulumu (4.5 adım 4) |
| İndirilen modeller ve dosya desenleri | Hücre 5, 5b | `download_models.py` |
| SheetSage2 ortamı | Hücre 5b: `TORCH` | `setup_windows.bat` (5/5 adım), Ubuntu kurulumu (4.5 adım 7) |
| Sunucu argümanları | Hücre 6: `server_args` | `start_model_server.sh` / `.bat`, Ubuntu systemd servisi (4.5) |
| Arayüzün başlatılması | Hücre 7 | `start_ui_mac.sh`, `start_ui_windows.bat` |
| Pakete giren dosyalar, zip adı | Hücre 3 (`BUNDLE`, `/content/yue-ui`) | `make_bundle.sh` |
| Kullanıcıya yönelik anlatım | Markdown hücreleri | `COLAB_YOL_HARITASI.md`, `README.md` |

**Yeni bir notebook varyantı** (ör. farklı bir arayüz sürümünü mevcut kurulumu bozmadan aynı Drive'da denemek):

1. `make_bundle.sh`'den türetilmiş ayrı bir paket betiği (`make_bundle_<ad>.sh`) ve ayrı zip adı
   (`yue2-colab-<ad>.zip`). Ana paket (`yue2-colab.zip`) olduğu gibi kalır.
2. Notebook kopyasında hücre 3'teki `BUNDLE` ve `DRIVE_BUNDLE` yeni zip adına çevrilir. Zip açıldıktan sonra
   içeriğinde varyanta özgü bir şey aranır, yoksa zip silinip hata verilir; böylece Drive'daki yanlış zip sessizce
   kullanılmaz.
3. Hücre 2'de `OUTPUTS` ayrı bir Drive klasörüne çevrilir (`yue2_outputs_<ad>`); geçmişler karışmaz.
4. `YUE2_CACHE` (`yue2_cache`) ortak kalır: ağırlıklar ve ortamlar yeniden inmez.
5. İlk markdown hücresine hangi zip'i kullandığı ve ana notebook'tan farkı yazılır.

**Web arayüzünü Colab'a almak** (yapılmadı, denenmedi): `make_bundle.sh`'ye `yue-ui/frontend` eklenmeli; hücre 7'ye
alternatif olarak `YUE2_PY /content/yue-ui/frontend/server.py --port 7870` başlatılıp
`eval_js("google.colab.kernel.proxyPort(7870)")` linki verilmeli. Web arayüzündeki adresler göreli olduğu için proxy
alt yolunda çalışması beklenir; Colab'da doğrulanmalı.

---

## 7. Çıktılar, önbellekler, loglar

| Yer | İçerik |
|---|---|
| `outputs/<iş-id>/` | `audio.mp3` (arayüzde çalan), `audio.flac` (48 kHz), `score.abc`, `semantic.npy`, `latent.npy`, `request.json`, `config.json`, `result.json`, `job.json`. İş id'si `YYYYMMDD-HHMMSS-xxxxxx` |
| `outputs/uploads/<sha256[:32]>/` | `source.<uzantı>` ve `analysis/` (`melody.wav`, `score.abc`, `lyrics.txt`, `result-<anahtar>.json`) |
| `outputs/mock/` | Mock sunucunun işleri |
| `logs/` | `nohup` ile başlatılan sunucuların ve `frontend/start.sh --mock`'un mock sunucusunun logları. Betikler normalde terminale yazar |
| `$TMPDIR/yue2-ui/` | Gradio'nun indirdiği ses dosyaları ve proje kayıtları |
| Colab | `/content/yue2_outputs` (ya da `USE_DRIVE` ile `MyDrive/yue2_outputs`), `/content/logs/` |

- **Analiz önbelleği:** Anahtar, dosyanın SHA-256'sı + analiz ayarları + `ANALYSIS_VERSION`. Önbellekteki analiz GPU
  istemez, anında döner. `cover.py`, `lyrics.py` ya da `sheetsage*.py` çıktısını değiştiren her değişiklikte
  [ANALYSIS_VERSION](../server/yue2_server.py#L39) bir artırılmalı; yoksa eski sonuçlar gelmeye devam eder.
- **Süre tahminleri:** `/stats`, bu cihazdaki tamamlanmış işlerin (yalnızca 32 ve 16 adımlı) "işlem sn / ses sn"
  medyanını verir. Cihazdaki ilk üretimden önce tahmin gösterilmez; yalnızca Apple MPS için ölçülmüş yedek değer var
  ([REFERENCE_RATES](../server/yue2_server.py#L42): 6.5 normal, 5.0 hızlı).

---

## 8. Testler

| Ne | Komut | Gerekli |
|---|---|---|
| Web arayüzü mantığı (`logic.js`, 25 test) | `./frontend/test.sh logic` | Node 18+ |
| Web sunucusu + `/api` vekili, mock sunucuyla uçtan uca (16 test) | `./frontend/test.sh server` | `yue-ui/venv` |
| Headless Chrome'da tüm akış (12 adım) | `./frontend/test.sh e2e` | Chrome, Node 22+ |
| Hepsi | `./frontend/test.sh` | |

- **Model sunucusu ve Gradio arayüzü için otomatik test yok.** Doğrulama mock modu ve `curl` ile yapılır
  (örnekler: [yapay-zeka-model-entegrasyonu.md §8.7](yapay-zeka-model-entegrasyonu.md#adım-8-modeli-projeye-bağla-model-sunucusu)).
- `frontend/public/js/logic.js`, Gradio fonksiyonlarının birebir portudur ve aynı girdilerle aynı metinleri üretir.
  Gradio mantığı değişirse `logic.js` de güncellenip testler çalıştırılmalı.

---

## 9. Bilinen sorunlar ve riskler

| Konu | Durum / öneri |
|---|---|
| Windows + RTX 3050 | Hiç çalıştırılmadı. İlk çalıştırmada beklenmeyen hata çıkabilir (FlashAttention, sürücü, yollar) |
| Ubuntu | Hiç çalıştırılmadı, kurulum betiği yok. `start_model_server.sh` `--budget auto` vermez; Linux'ta paylaşımlı GPU belleği olmadığı için 8–12 GB kartta uzun şarkı + CFG bellek hatası verir (4.6) |
| Mac'te uzun şarkılar | ~6 dk'lık şarkı sentezde OOM; saatlerce hesap boşa gidebilir |
| Cover · Serbest'te eşlik orijinale benziyor | Önerilip yapılmayanlar: CFG 1.5–2.5 + orijinalden farklı enstrüman etiketleri; "Enstrüman ezgilerini gönderme" seçeneği (`V: Ins` satırlarını SheetSage2'nin boş biçimi `Z<n>\|` ile değiştirmek). Hedef sık değiştiği için önce proje sahibiyle karar verilmeli |
| Basit analizde tempo | 4/3 ya da 2 kat yanılabiliyor; ölçü başları tahmin edilmiyor (4/4 varsayılıyor) |
| Transkripsiyon hataları | Notadaki hatalar cover'a geçer; resmi kılavuz notanın elle kontrolünü öneriyor |
| Söz tanıma + GPU paylaşımı Colab'da | Denenmedi; bellek değerleri (YuE2 ~7.3 GB, Demucs + Whisper ~5–6 GB) tahmin |
| Hızlı sentez (16 adım) ve FP8 | Kaliteye etkisi karşılaştırmalı dinlenmedi |
| Gradio ilerleme mesajı | SheetSage2 seçiliyken de "vokal ayırma + melodi çıkarma" yazıyor ([yue2_ui.py:766](../source/yue2_ui.py#L766)); metin Basit analizden kalma |
| Herkese açık site | Siteyi açabilen herkes GPU kuyruğuna iş ekler ve Geçmiş'te tüm üretimleri (sözler dahil) görür; yüklemeler diskte kalır. Parola HTTP Basic: TLS'li ters vekil (Caddy, nginx, Cloudflare Tunnel) şart |
| Belgelerde eskimiş yerler | `ENTEGRASYON.md`'deki mimari şeması SheetSage2, Whisper ve web arayüzünü göstermiyor; `ENTEGRASYON.md` §9, `yapay-zeka-model-entegrasyonu.md` sonu ve `COLAB_YOL_HARITASI.md`'nin "Bilinen sınırlar"ının ilk maddesi "Colab'da çalıştırılmadı" diyor, oysa L4 ölçümü var. Bu belgedeki bilgiler 16 Ekim itibarıyla günceldir |

---

## 10. Açık işler

Öncelik sırasıyla:

| # | İş | Tahmini | Nereler değişir |
|---|---|---|---|
| 1 | Commit edilmemiş belgeleri (bölüm 2) depoya almak | dakikalar | `docs/` |
| 2 | Güncel koddan Colab paketini üretip bir oturumda Text to Song + Cover denemek (6.2'deki not) | yarım gün | `colab/` |
| 3 | Windows'ta RTX 3050 ile ilk deneme ve ölçüm (`--budget auto` satırı, VRAM, süre) | 1 gün | Kod değişikliği beklenmiyor; [resolve_budget](../server/yue2_server.py#L544) |
| 4 | Colab'da Sözleri tanı ve GPU paylaşımı testi | yarım gün | [make_room](../server/yue2_server.py#L595), [gpu_for_analysis](../server/yue2_server.py#L483) |
| 5 | Cover eşliğini türe çevirme (9'daki seçenekler) | karar + 1 gün | `source/yue2_ui.py`, `frontend/public/js/` |
| 6 | Herkese açık site: Redis kuyruğu; kuyruk doluysa istek baştan reddedilir | 1.5 gün | [Server.queue](../server/yue2_server.py#L424), [_worker](../server/yue2_server.py#L453) |
| 7 | API'de sıra numarası yerine `eta_seconds`/`progress`; arayüzde kuyruk+yükleme+nota planı yerine tek "Hazırlanıyor" durumu, teknik panel yalnızca yönetici modunda | 1 gün | [Job.to_dict](../server/yue2_server.py#L111), [rates](../server/yue2_server.py#L501), `frontend/public/js/logic.js` (`STAGE_LABELS`), `output.js` (`renderJob`) |
| 8 | Hızlı sentez ve FP8 için kalite dinlemesi | yarım gün | – |
| 9 | Eskimiş belgeleri güncellemek (9'daki liste) | yarım gün | `docs/`, `colab/COLAB_YOL_HARITASI.md` |
| 10 | Model sunucusu için mock tabanlı otomatik testler | 1 gün | yeni `server/tests/` |
| 11 | Ubuntu'da ilk kurulum (4.5); adımlar tutarsa `setup_linux.sh` betiğine çevirmek, `start_model_server.sh`'ye `--budget auto` eklemek (Mac'te etkisiz: GPU bulamayınca 24 GiB kalır), 4.6'yı karşılaşılan hatalarla güncellemek | 1 gün | `setup_linux.sh` (yeni), `start_model_server.sh` |

Kuyruk tasarımında alınmış karar: kullanıcıya sıra numarası ("önünüzde N şarkı var") gösterilmeyecek; yalnızca
şarkının ne zaman hazır olacağı (tahmini süre) gösterilecek. GPU başına tek iş çalıştığı için yük altındaki asıl sorun
bekleme süresidir.

---

## 11. Belge haritası

| Belge | Ne zaman okunur |
|---|---|
| [README.md](../README.md) | Kullanım: başlatma, Windows kurulumu, Text to Song, Cover, model ayarları, çıktılar |
| Bu belge | İlk gün: durum, kurulum, modlar, Colab, açık işler |
| [ENTEGRASYON.md](ENTEGRASYON.md) | Mimari kararlar, YuE-UI'dan neden ayrılındı, karşılaşılan sorunlar, neden 24 GB / NVIDIA |
| [model-calisma-prensibi.md](model-calisma-prensibi.md) | Altı modelin girdileri/çıktıları, modeller arası veri akışı, resmi cover tarifiyle ve NoizAI sitesiyle karşılaştırma |
| [yapay-zeka-model-entegrasyonu.md](yapay-zeka-model-entegrasyonu.md) | Model kurulumu adım adım, `yue2 doctor`, ilk üretim, CUDA çekirdekleri ve MPS farkı |
| [rtx3050-windows-raporu.md](rtx3050-windows-raporu.md) | `--phase-offload` tasarımı, bellek dökümü, Windows'ta yapılacaklar |
| [gun-gun-gelistirme-plani.md](gun-gun-gelistirme-plani.md) | Projenin 35 günlük fazlara bölünmüş geliştirme sırası; her işin koddaki karşılığı (fonksiyon bağlantılarıyla) |
| [../colab/COLAB_YOL_HARITASI.md](../colab/COLAB_YOL_HARITASI.md) | Colab'ı kullanıcı olarak çalıştırma, sorun giderme |
| [../frontend/README.md](../frontend/README.md) | Web arayüzü: başlatma, testler, Gradio'yla farklar, yayınlama |
| `../../staj-konu-aciklamalari.md` (depo dışı) | Projede geçen kavramların (KV önbelleği, flow matching, CFG, pYIN, WER...) gün gün açıklaması |

---

## 12. Sözlük

| Terim | Anlamı |
|---|---|
| **ABC** | Metin tabanlı nota yazımı. YuE2'nin nota biçimi: `L:1/32`, `V: Vocal` / `V: Ins` sesleri, `% bölüm` satırları, tırnak içinde akorlar (`"Gm"`) |
| **CoT** | "Chain of thought": modelin şarkıdan önce nota planı yazması (`full`, `melody`, `off`) |
| **AR / NAR** | YuE2-3B'nin iki yolu. AR (otoregresif) token'ları tek tek üretir (nota planı, ses token'ları); NAR ses sentezini yapar. 28 katmanın her birinde ikisi de var |
| **Ses token'ı / latent** | Sesin sıkıştırılmış temsili; saniyede ~25. NAR token'ları 64 boyutlu latent'lere, VAE latent'leri 48 kHz stereo sese çevirir |
| **ODE adımı** | Ses sentezindeki flow matching çözümünün adım sayısı (32; hızlı sentezde 16) |
| **VAE** | Variational autoencoder. YuE2'de yalnızca çözücüsü kullanılır: latent → ses |
| **CFG** | Classifier-free guidance. Modeli koşullu ve koşulsuz iki dalda çalıştırıp farkı büyütür; etiketlere daha sıkı uyum, iki kat bellek ve süre |
| **KV önbelleği** | Üretilen token'ların dikkat geçmişi; token başına ~112 KiB. 9000 token ≈ 1 GiB |
| **Kaynak ayrıştırma** | Karışık kaydı vokal/davul/bas/diğer kanallarına ayırma (Demucs) |
| **Mock** | Model yerine deneme tonu üreten sahte sunucu; arayüz geliştirme ve testler için |
