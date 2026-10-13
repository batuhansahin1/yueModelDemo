# Yapay Zeka Model Entegrasyonu: YuE2

Bu belge, YuE2 müzik üretim modelinin bu projeye nasıl kurulduğunu ve çalışır hale getirildiğini adım adım anlatır.
**Arayüz (Gradio) kapsam dışıdır.** Belge model ortamını, ağırlıkları, doğrulamayı, ilk üretimi ve modeli HTTP
üzerinden sunan model sunucusunu kapsar. İkinci bölüm, CUDA çekirdeklerinin ne olduğunu ve YuE2'nin NVIDIA
GPU'larda neden daha hızlı çalıştığını koddan örneklerle açıklar.

Arayüz ve cover akışı için [ENTEGRASYON.md](ENTEGRASYON.md), Colab için
[../colab/COLAB_YOL_HARITASI.md](../colab/COLAB_YOL_HARITASI.md) belgesine bakın.

---

## Kurulum

### Özet

| | |
|---|---|
| Model | YuE2-3B (`m-a-p/YuE2-3B`, ~3.6 milyar parametre, BF16) |
| Ses çözücü | YuE2-VAE (`m-a-p/YuE2-Vae`, 48 kHz stereo) |
| Çalıştırma kodu | `yue2-infer 0.1.5` (Python wheel, 66 KB, ağırlık içermez) |
| Python | 3.12.10 (python.org sürümü), sanal ortam `yue_ortam/` |
| Test edilen donanım | Apple M4, 24 GB birleşik bellek, MPS |
| Çalışma şekli | Ağırlıklar yerelde, tamamen çevrimdışı |

Kurulum sonunda proje klasörü şöyle görünür:

```
yueModel/
├── yue2_infer-0.1.5-py3-none-any.whl   # YuE2 çalıştırma kodu
├── yue_ortam/                          # Model için Python 3.12 sanal ortamı
├── models/
│   ├── YuE2-3B/                        # Ana model (7.3 GB)
│   └── YuE2-Vae/                       # Ses çözücü (0.5 GB)
└── yue-ui/
    ├── server/yue2_server.py           # Modeli HTTP API olarak sunan sunucu
    └── start_model_server.sh           # Sunucuyu çevrimdışı modda başlatan betik
```

### Adım 1: Ön koşullar

- **Python 3.12.** yue2 en az Python 3.10 istiyor. Kurulumda python.org'dan kurulan 3.12.10 kullanıldı.
- **Disk:** Ağırlıklar için ~8 GB, Python paketleri (PyTorch dahil) için ~2-3 GB.
- **Bellek:** Model 24 GB'ı hedefliyor ama daha azıyla da çalışıyor (bkz. [Adım 9](#adım-9-bellek-ve-hız-ayarları)).
- **Donanım:** NVIDIA GPU (CUDA), Apple Silicon (MPS) veya CPU. yue2 cihazı kendisi seçiyor: önce CUDA,
  sonra MPS, en son CPU ([pipeline.py:134-135](../../yue_ortam/lib/python3.12/site-packages/yue2/pipeline.py#L134-L135)).

### Adım 2: yue2 wheel'ini indir

YuE2'yi çalıştıran kod, Hugging Face'teki model deposunda ayrı bir wheel dosyası olarak dağıtılıyor. Wheel, model
ağırlıklarıyla aynı revizyondan (`c044757`) proje köküne indirildi:

```bash
cd ~/Desktop/yueModel
hf download m-a-p/YuE2-3B yue2_infer-0.1.5-py3-none-any.whl --local-dir .
```

> `hf` komutu `huggingface_hub` paketiyle gelir. Henüz bir ortam yoksa sistem Python'una geçici olarak
> `pip install huggingface_hub` ile kurulabilir ya da dosya tarayıcıdan indirilebilir.

### Adım 3: Model için sanal ortam oluştur

yue2 tüm bağımlılıklarının sürümünü sabitliyor. Başka projelerin paketleriyle çakışmaması için modele ayrı bir
sanal ortam açıldı:

```bash
python3.12 -m venv yue_ortam
source yue_ortam/bin/activate
pip install --upgrade pip
```

### Adım 4: yue2'yi kur

```bash
pip install ./yue2_infer-0.1.5-py3-none-any.whl
```

Wheel şu sürümleri sabit olarak kuruyor. Bu sürümler **elle değiştirilmemeli**; yue2 bunlarla test edilmiş:

| Paket | Sürüm | Görevi |
|---|---|---|
| `torch` | 2.10.0 | Tensör hesapları, CUDA/MPS erişimi |
| `transformers` | 4.57.6 | Modelin yüklenmesi (`YuE2ForCausalLM`) |
| `huggingface-hub` | 0.36.2 | Ağırlıkların indirilmesi ve çözümlenmesi |
| `safetensors` | 0.7.0 | Ağırlık dosyası formatı |
| `tiktoken` | 0.12.0 | Metin tokenizer'ı (`qwen.tiktoken`) |
| `numpy` | 2.2.6 | Dizi işlemleri |
| `soundfile` | 0.13.1 | FLAC/WAV/MP3 yazma (libsndfile) |
| `accelerate` | 1.13.0 | Düşük bellekle model yükleme |

İsteğe bağlı ekstralar: `[serve]` (FastAPI + uvicorn), `[fast]` (vLLM + Triton, yalnızca NVIDIA),
`[bench]` (NVIDIA bellek/güç ölçümü). Kurulumda yalnızca sunucu için gerekenler eklendi (Adım 8).

Kurulumu doğrula:

```bash
yue_ortam/bin/python -c "import yue2, torch; print(yue2.__version__, torch.__version__, torch.backends.mps.is_available(), torch.cuda.is_available())"
# 0.1.5 2.10.0 True False      (Mac M4'te)
```

### Adım 5: Model ağırlıklarını indir

Ağırlıklar Hugging Face önbelleğine değil, doğrudan projenin `models/` klasörüne bir kez indirildi. Böylece hangi
dosyaların kullanıldığı görünür oluyor ve sonraki çalıştırmalarda internete çıkmaya gerek kalmıyor.

```bash
yue_ortam/bin/hf download m-a-p/YuE2-Vae --local-dir models/YuE2-Vae
yue_ortam/bin/hf download m-a-p/YuE2-3B  --local-dir models/YuE2-3B
```

İndirilen dosyalar:

| Klasör | Dosya | Boyut | Ne işe yarar |
|---|---|---|---|
| `YuE2-3B` | `model.safetensors` | 7.26 GB | Ana modelin BF16 ağırlıkları |
| | `config.json` | 1 KB | Mimari: 28 katman, 2048 gizli boyut, 16 sorgu / 8 KV kafası, 24.576 token bağlam |
| | `qwen.tiktoken` | 2.5 MB | Metin tokenizer'ı |
| | `yue2_generation_config.json` | 1 KB | Varsayılan örnekleme ayarları (ABC ve şarkı aşamaları, 32 ODE adımı) |
| | `weights_manifest.json` | 1 KB | Ağırlık dosyasının boyutu ve SHA-256 özeti |
| `YuE2-Vae` | `model.safetensors` | 0.53 GB | Latent'leri 48 kHz sese çeviren VAE |
| | `config.json`, `weights_manifest.json` | | Yapılandırma ve SHA-256 özeti |

> `--local-dir` kullanıldığında `huggingface_hub` her klasörde `.cache/huggingface/` altına indirme meta
> verisini bırakır; bu normaldir.

### Adım 6: Kurulumu doğrula

yue2 ile gelen `doctor` komutu paket sürümlerini, GPU'ları ve ağırlıkların bütünlüğünü kontrol ediyor:

```bash
yue_ortam/bin/yue2 doctor --model models/YuE2-3B --vae models/YuE2-Vae --offline --verify-hashes
```

- `dependencies_ready: true` olmalı.
- `weights` bölümünde iki dosyanın SHA-256 özeti `weights_manifest.json` ile eşleşmeli:
  - YuE2-3B: `1d55c42c…a1e59e9`
  - YuE2-Vae: `807ce9d5…7751346`
- Mac'te `mps_available: true`, `cuda: []` görünür. NVIDIA makinede `cuda` listesinde kartın adı, belleği ve
  compute capability değeri yer alır.

> `--model` ve `--vae` açıkça verilmeli. Verilmezse CLI ağırlıkları paketin kurulu olduğu yerde arıyor, bulamazsa
> Hugging Face'ten indirmeye çalışıyor ([cli.py:13-24](../../yue_ortam/lib/python3.12/site-packages/yue2/cli.py#L13-L24)).

Pipeline da her yüklemede aynı doğrulamayı yapıyor ("Verifying model files", M4'te ~3.7 sn). Bozuk ya da yarım
inmiş bir ağırlık dosyasıyla model hiç yüklenmiyor.

### Adım 7: İlk üretim (komut satırı)

Ağırlıklar indikten sonra Hugging Face'e hiçbir istek gitmemesi için şu ortam değişkenleri kullanıldı. MPS'te henüz
desteklenmeyen birkaç işlemin CPU'ya düşebilmesi için `PYTORCH_ENABLE_MPS_FALLBACK=1` de ekleniyor:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

yue_ortam/bin/yue2 generate --offline \
    --model models/YuE2-3B --vae models/YuE2-Vae \
    --style "Turkish, acoustic pop, warm piano, male vocal" \
    --lyrics-file sozler.txt --cot full --seed 42 \
    --id ilk_sarki --output runs
```

Sözler `[verse]`, `[chorus]` gibi bölüm etiketleriyle yazılıyor. Çıktı `runs/ilk_sarki/` klasörüne gidiyor:
`audio.flac` (48 kHz, 24 bit), `score.abc` (modelin yazdığı nota), `semantic.npy`, `latent.npy`,
`request.json`, `config.json` ve tüm dosyaların SHA-256 özetlerini içeren `result.json`.

Model bir şarkıyı dört aşamada üretiyor:

| Aşama | Ne olur |
|---|---|
| 1. Nota planı (ABC) | Model önce tempo, ton, akor ve bölümleri içeren bir ABC nota yazar |
| 2. Şarkı token'ları | Nota + sözler + stil koşulunda ses token'ları üretilir (sesin saniyesi başına ~25 token) |
| 3. Ses sentezi (NAR) | Aynı model, flow-matching ile token'lardan sürekli latent'ler üretir (32 ODE adımı) |
| 4. Ses çözme (VAE) | Latent'ler 48 kHz stereo dalga formuna çevrilir |

`--cot` seçenekleri: `full` (akorlu ABC), `melody` (akorsuz ABC), `off` (nota yok, doğrudan ses token'ları).

### Adım 8: Modeli projeye bağla (model sunucusu)

Modelin projenin geri kalanından kullanılabilmesi için [server/yue2_server.py](../server/yue2_server.py) yazıldı.
Sunucu `yue2.YuE2Pipeline`'ı yükleyip küçük bir HTTP API olarak sunuyor. Böylece model, kendi sabit sürümlü
ortamında ayrı bir süreç olarak çalışıyor ve onu çağıran taraf PyTorch kurmak zorunda kalmıyor.

**8.1 Sunucu bağımlılıkları.** yue2'nin `[serve]` ekstrasıyla aynı sürümler `yue_ortam`'a eklendi.
[start_model_server.sh](../start_model_server.sh) bunları eksikse kendisi kuruyor:

```bash
yue_ortam/bin/pip install "fastapi==0.136.1" "uvicorn==0.46.0"
```

**8.2 Pipeline'ın yüklenmesi.** Sunucu modeli Python API üzerinden, yerel klasörlerden ve çevrimdışı yüklüyor
([yue2_server.py:166-179](../server/yue2_server.py#L166-L179)):

```python
from yue2.pipeline import YuE2Pipeline

pipe = YuE2Pipeline.from_pretrained(
    "models/YuE2-3B", vae="models/YuE2-Vae",
    device="auto",              # cuda > mps > cpu
    memory_budget_gib=24,       # bellek bütçesi
    backend="torch",            # CUDA'da CUDA graph kullanır
    quantization="none",        # "fp8" yalnızca NVIDIA compute 8.9+
    offload_ar=False,
    local_files_only=True,      # internete çıkma
    vae_core_frames=1024)       # bütçe <= 12 GB ise 512
```

**8.3 Aşama aşama çalıştırma.** `pipe(...)` tek çağrıda bütün şarkıyı üretebiliyor. Sunucu ise ilerlemeyi
izleyebilmek ve her aşamada iptal edebilmek için dört aşamayı ayrı ayrı çağırıyor
([yue2_server.py:208-247](../server/yue2_server.py#L208-L247)):

```python
plan     = pipe.plan(request=request, cancelled=cancelled, on_token=on_token)        # 1. ABC
semantic = pipe.generate_semantic(plan, cancelled=cancelled, on_token=on_token)      # 2. Token'lar
latents  = pipe.synthesize(semantic, cancelled=cancelled)                            # 3. NAR
audio    = pipe.decode(latents)                                                      # 4. VAE
```

`on_token` geri çağrısı her üretilen token'ı sayıyor; `cancelled` geri çağrısı `True` dönünce yue2 o aşamayı
yarıda kesiyor. Sonuç `SongResult.save_artifacts()` ile yue2'nin kendi formatında kaydediliyor, ardından
libsndfile'ın yerleşik kodlayıcısıyla bir `audio.mp3` de yazılıyor (ffmpeg gerekmiyor).

**8.4 İş kuyruğu.** yue2'nin resmi runtime'ı aynı anda tek üretim destekliyor (`concurrency=1`). Bu yüzden
sunucu gelen işleri tek bir çalışan kuyruğunda sırayla işliyor. Her iş `outputs/<iş-id>/job.json` dosyasına
yazıldığı için sunucu yeniden başlasa da geçmiş kaybolmuyor.

**8.5 API.**

| Uç nokta | İşlev |
|---|---|
| `GET /health` | Cihaz, modelin bellekte olup olmadığı, yükleme hatası, kuyruk durumu |
| `POST /jobs` | Üretim işi başlatır (stil, sözler, CoT, seed, CFG, harici ABC, örnekleme, `ode_steps`) |
| `GET /jobs`, `GET /jobs/{id}` | İş listesi; tek işin aşaması, token sayıları, hatası ve sonucu |
| `POST /jobs/{id}/cancel` | Çalışan işi iptal eder |
| `GET /jobs/{id}/audio` | Üretilen MP3'ü döndürür |

**8.6 Başlatma.** Betik ağırlıkların varlığını kontrol ediyor, çevrimdışı ortam değişkenlerini ayarlıyor ve
sunucuyu yalnızca `127.0.0.1:7861` adresinde açıyor:

```bash
cd yue-ui
./start_model_server.sh --preload         # modeli başlangıçta belleğe yükle
./start_model_server.sh --mock            # modelsiz test: kısa bir deneme tonu üretir
YUE2_MODELS=/baska/yol ./start_model_server.sh   # ağırlıklar başka klasördeyse
```

**8.7 Uçtan uca test.** Önce `--mock` ile API akışı model olmadan doğrulandı. Ardından gerçek modelle:

```bash
curl -s 127.0.0.1:7861/health
# {"status":"ok","mode":"yue2","model_loaded":true,"device":"mps",...}

curl -s -X POST 127.0.0.1:7861/jobs -H 'Content-Type: application/json' \
     -d '{"style":"acoustic pop, female vocal","lyrics":"[verse]\n...\n[chorus]\n...","cot":"full","seed":42}'

curl -s 127.0.0.1:7861/jobs/<iş-id>                 # stage: abc → semantic → synthesize → decode → done
curl -s 127.0.0.1:7861/jobs/<iş-id>/audio -o sarki.mp3
```

İlk gerçek üretim Mac M4 üzerinde 49.5 saniyelik bir şarkıyı 5 dk 12 sn'de tamamladı.

### Adım 9: Bellek ve hız ayarları

| Ayar | Nerede | Etkisi |
|---|---|---|
| `--budget <GiB>` | sunucu, CLI | Bellek bütçesi. CUDA'da GPU bellek sınırı buna göre konuyor; ≤12 GB'ta VAE daha küçük parçalarla çözülüyor |
| `--offload-ar` | sunucu, CLI | Sentez sırasında belleği azaltır (16 GB kartlar için) |
| `--phase-offload` | sunucu, CLI | Her aşamada modelin yalnızca kullanılan yarısı GPU'da (token üretiminde AR, sentezde NAR), diğeri RAM'de. 8-12 GB kartlarda `--budget auto` açar |
| `ode_steps` | iş başına (`POST /jobs`) | Sentez adımı 32 yerine 16: M4'te sentez süresi sesin saniyesi başına 3.3 sn'den 2.0 sn'ye indi |
| Bellek boşaltma | sunucu varsayılanı | Kuyruk boşalınca ağırlıklar bellekten atılır, sonraki işte ~11 sn'de yüklenir. `--keep-loaded` bunu kapatır |
| `--device` | sunucu, CLI | `auto`, `cuda`, `mps`, `cpu` |
| `--backend` | sunucu, CLI | `torch` (varsayılan), `torch-eager`, `vllm` (yalnızca NVIDIA, `[fast]` ekstrası) |
| `--quantization fp8` | sunucu, CLI | Yalnızca NVIDIA compute capability 8.9+; kalite etkisi denenmedi, varsayılan kapalı |

### Mac M4 ölçümleri (49.5 sn şarkı)

| Aşama | Süre | Token |
|---|---|---|
| Model yükleme | 11 sn | |
| 1. Nota planı (ABC) | 32 sn | 488 |
| 2. Şarkı token'ları | 91 sn | 1238 |
| 3. Ses sentezi (32 adım) | 165 sn | |
| 4. Ses çözme (VAE) | 12 sn | |
| **Toplam** | **5 dk 12 sn** (~6.3× gerçek zaman) | |

Token üretimi başlangıçta ~14 token/sn, uzun şarkılarda 6.7 token/sn'ye düştü. 4 dakikalık bir denemede
birleşik bellek dolduğu için 7.8 GB swap kullanıldı.

### Kurulumda karşılaşılan sorunlar

| Sorun | Sebep | Çözüm |
|---|---|---|
| Sunucu modeli "hazır" gösteriyordu ama ilk iş uzun sürüyordu | Dosya doğrulaması ile ağırlıkların belleğe yüklenmesi ayırt edilmiyordu | `model_loaded` yalnızca ağırlıklar gerçekten bellekteyken `true` |
| Üretim çok yavaş, Mac ısınıyor | MPS'te CUDA'ya özel hızlandırmalar yok (bkz. sonraki bölüm); uzun bağlam; swap | Hızlı sentez (16 adım), daha kısa şarkılar, işten sonra bellek boşaltma |
| Sunucunun editörle birlikte kapanması | Terminal kapanınca süreç sonlanıyordu | `nohup` ile başlatıldı; loglar `yue-ui/logs/model_server.log` |

---

## CUDA Çekirdeği ve NVIDIA'nın Avantajları

### CUDA ve CUDA çekirdeği nedir?

**CUDA**, NVIDIA'nın GPU'ları genel amaçlı hesaplama için programlamaya yarayan platformudur. Bir **CUDA
çekirdeği (kernel)**, GPU üzerinde aynı anda binlerce iş parçacığıyla (thread) çalışan tek bir fonksiyondur.
Örneğin bir matris çarpımı ya da bir dikkat (attention) hesabı bir çekirdektir.

Bir çekirdeğin çalışma şekli:

1. CPU, GPU'ya "şu çekirdeği şu verilerle çalıştır" komutunu gönderir (**kernel launch**).
2. GPU, işi binlerce küçük parçaya böler ve bunları onlarca hesaplama biriminde (streaming multiprocessor, SM)
   paralel çalıştırır.
3. Sonuç GPU belleğinde kalır ve sonraki çekirdek onu doğrudan kullanır.

PyTorch'ta yazılan her `matmul`, `softmax`, `layernorm` aslında bir ya da birkaç CUDA çekirdeğine dönüşür.
Bu çekirdekler NVIDIA'nın hazır kütüphanelerinden gelir:

| Kütüphane | Ne sağlar |
|---|---|
| **cuBLAS** | Matris çarpımları (modeldeki doğrusal katmanların neredeyse tamamı) |
| **cuDNN** | Dikkat, evrişim (convolution), normalizasyon gibi sinir ağı çekirdekleri |
| **FlashAttention** | Dikkati, ara matrisi belleğe yazmadan tek çekirdekte hesaplayan optimize çekirdek |
| **CUDA Graphs** | Bir çekirdek dizisini bir kez kaydedip tek komutla tekrar çalıştırma |
| **NCCL / NVML** | Çoklu GPU iletişimi; bellek, güç ve sıcaklık ölçümü |

### Tensor Core'lar ve sayı formatları

Volta mimarisinden (2017) bu yana NVIDIA GPU'larda **Tensor Core** adlı, yalnızca matris çarpımı yapan özel
birimler var. Yapay zeka modellerinin hesabının büyük kısmı matris çarpımı olduğu için hızın asıl kaynağı bunlar.
Tensor Core'lar düşük hassasiyetli formatlarda çok daha hızlı çalışıyor:

| Format | Bayt | Desteklenen NVIDIA nesli | YuE2'de kullanımı |
|---|---|---|---|
| FP32 | 4 | Tüm GPU'lar | VAE (ses çözme) |
| **BF16** | 2 | Ampere (compute 8.0) ve sonrası, donanımda | Ana modelin tamamı; CUDA'da zorunlu |
| **FP8 (E4M3)** | 1 | Ada Lovelace / Hopper (compute 8.9+) | İsteğe bağlı, token üretimindeki doğrusal katmanlar |

BF16, FP32 ile aynı sayı aralığını yarı bellekle tutuyor. 3.6 milyar parametre BF16'da 7.3 GB, FP32'de olsaydı
~14.5 GB olurdu. FP8 ağırlık belleğini bir kez daha yarıya indiriyor ve uygun kartlarda matris çarpımını
hızlandırıyor.

### YuE2 CUDA'yı nerede kullanıyor?

yue2'nin kodu cihaz CUDA olduğunda ayrı, daha hızlı yollara giriyor. Aşağıdakilerin hepsi koddan doğrulandı.

**1. Token üretiminde CUDA Graph** ([cuda_graph.py](../../yue_ortam/lib/python3.12/site-packages/yue2/cuda_graph.py))

Model, şarkının 2. aşamasında token'ları **tek tek** üretiyor: 50 saniyelik bir şarkı için ~1.250 token ve her
token için 28 katmanın tamamı yeniden çalışıyor. Her katmanda normalizasyon, Q/K/V projeksiyonları, döndürme
(RoPE), KV cache'e yazma, dikkat, çıkış projeksiyonu ve MLP bulunuyor. Yani **tek bir token için yüzlerce küçük
çekirdek** çalıştırılıyor. Bu çekirdekler o kadar küçük ki GPU'nun hesap süresi, CPU'nun her birini başlatma
süresinden kısa kalıyor; GPU zamanının çoğu bir sonraki komutu beklemekle geçiyor.

CUDA Graph bu sorunu çözüyor. yue2 bir token adımını bir kez kaydediyor (`torch.cuda.CUDAGraph`), sonraki her
token için kaydı tek komutla tekrar oynatıyor (`graph.replay()`). Yüzlerce kernel launch tek bir launch'a
iniyor. Kod bunu açıkça yalnızca CUDA'ya izin veriyor:

```python
# cuda_graph.py:66-67
if capture and self.device.type != "cuda":
    raise ValueError("CUDA graphs require a CUDA model; capture=False is for diagnostics")
```

MPS ve CPU'da her token Python'dan ayrı ayrı çalışıyor (`execution: "eager"`). M4'te token hızının 6.7-14
token/sn'de kalmasının ana nedeni bu.

**2. FlashAttention ve cuDNN dikkat çekirdekleri** ([cuda_graph.py:74-86](../../yue_ortam/lib/python3.12/site-packages/yue2/cuda_graph.py#L74-L86))

CUDA'da yue2 dikkat hesabı için önce PyTorch'un yerleşik **değişken uzunluklu FlashAttention** çekirdeğini,
o yoksa **cuDNN attention**'ı seçiyor. FlashAttention, büyük dikkat matrisini GPU belleğine yazmadan çipin
içindeki hızlı bellekte parça parça hesaplıyor; bu hem bellekten hem zamandan kazandırıyor. Diğer cihazlarda
genel amaçlı SDPA kullanılıyor.

**3. Ses sentezinde tek blok dikkat** ([nar.py:70-78](../../yue_ortam/lib/python3.12/site-packages/yue2/nar.py#L70-L78))

3. aşamada CUDA'da tüm sorgular tek blok halinde, birleşik (fused) SDPA çekirdeğiyle işleniyor. MPS'te ise
sorgular 256'lık bloklara bölünüyor ve modelin gruplu dikkat yapısı (16 sorgu kafası / 8 KV kafası) desteklenmediği
için anahtar ve değer tensörleri her adımda kopyalanarak çoğaltılıyor (`repeat_interleave`). Bu ek bellek
kopyası 32 ODE adımının her birinde tekrar ediyor. M4'te sentezin en uzun aşama olmasının (165 sn) nedeni bu.

**4. FP8 Tensor Core çarpımı** ([quantization.py](../../yue_ortam/lib/python3.12/site-packages/yue2/quantization.py))

`--quantization fp8` seçildiğinde token üretimindeki doğrusal katmanların ağırlıkları FP8'e çevriliyor ve çarpım
`torch._scaled_mm` ile doğrudan cuBLAS'ın FP8 Tensor Core çekirdeğinde yapılıyor. Kod compute capability
8.9'un altını reddediyor. Sentez aşamasından önce orijinal BF16 ağırlıklar birebir geri yükleniyor, yani FP8
sadece token üretimini etkiliyor.

**5. vLLM arka ucu** ([fast.py](../../yue_ortam/lib/python3.12/site-packages/yue2/fast.py))

`--backend vllm` ile token üretimi vLLM'e devrediliyor. vLLM, PagedAttention ve Triton ile yazılmış CUDA
çekirdekleri kullanan, büyük dil modelleri için optimize bir çıkarım motoru. `[fast]` ekstrasıyla kuruluyor ve
yalnızca NVIDIA'da çalışıyor.

**6. GPU belleği yönetimi** ([pipeline.py:158-165](../../yue_ortam/lib/python3.12/site-packages/yue2/pipeline.py#L158-L165))

CUDA'da pipeline BF16 desteğini kontrol ediyor ve `torch.cuda.set_per_process_memory_fraction` ile kendi
bellek tavanını koyuyor: `min(bütçe − 2 GB, kart belleği − 2 GB)`. GPU'nun kendi belleği (VRAM) olduğu için
model işletim sistemiyle yarışmıyor. Mac'teki birleşik bellekte ise CPU ve GPU aynı 24 GB'ı paylaşıyor; bellek
dolunca veri diske (swap) taşınıyor ve üretim yavaşlıyor.

> Not: yue2 tekrarlanabilirlik için TF32'yi bilerek kapatıyor ve cuDNN'i deterministik modda çalıştırıyor
> ([pipeline.py:139-144](../../yue_ortam/lib/python3.12/site-packages/yue2/pipeline.py#L139-L144)). Aynı seed
> aynı şarkıyı üretiyor; bunun bedeli CUDA'daki birkaç küçük hız optimizasyonundan vazgeçmek.

### CUDA ve Apple MPS karşılaştırması

| Özellik | NVIDIA CUDA | Apple MPS (bu kurulum) |
|---|---|---|
| Token adımı çalıştırma | CUDA Graph: kaydet, tek komutla oynat | Eager: her çekirdek Python'dan ayrı başlatılır |
| Dikkat çekirdeği | FlashAttention (varlen) / cuDNN | Genel SDPA; gruplu dikkatte K/V kopyalanır |
| BF16 matris çarpımı | Tensor Core, donanımda | Destekleniyor, hızlandırma sınırlı |
| FP8 | compute 8.9+ kartlarda | Yok |
| vLLM | Var | Yok |
| Bellek | Ayrı VRAM, süreç başına tavan | CPU ile paylaşılan birleşik bellek, swap riski |
| Bazı işlemler | Hepsi GPU'da | Bazıları CPU'ya düşüyor (`PYTORCH_ENABLE_MPS_FALLBACK=1`) |

### NVIDIA'nın genel avantajları

- **Ekosistem.** PyTorch, transformers, vLLM, FlashAttention gibi kütüphaneler önce CUDA için yazılıyor ve en
  çok CUDA'da test ediliyor. Yeni optimizasyonlar diğer platformlara ya geç geliyor ya hiç gelmiyor. YuE2'nin
  kendisi bunun örneği: hızlandırmalarının hepsi CUDA'ya özel.
- **Olgun çekirdek kütüphaneleri.** cuBLAS ve cuDNN her GPU nesli için elle optimize edilmiş çekirdekler içeriyor.
  PyTorch işlemleri otomatik olarak bunlara yönlendiriliyor; geliştiricinin bir şey yapmasına gerek kalmıyor.
- **Tensor Core ve düşük hassasiyet.** BF16 ve FP8 donanım desteği, hem belleği yarıya indiriyor hem de matris
  çarpımını katlarca hızlandırıyor.
- **Ayrılmış bellek (VRAM).** Model, işletim sistemi ve diğer uygulamalarla bellek paylaşmıyor. HBM/GDDR
  bellek bant genişliği, token üretimi gibi belleğe bağlı işlerde belirleyici.
- **Ölçeklenme.** Aynı kod tek bir dizüstü GPU'sundan bulut GPU'larına (L4, A100, H100) değişmeden taşınıyor.
  NCCL ile çoklu GPU, NVML ile izleme hazır geliyor.
- **Bulutta erişim.** Colab, AWS, GCP gibi servislerin GPU seçeneklerinin neredeyse tamamı NVIDIA.

### YuE2 için NVIDIA kartları

| Kart | VRAM | Compute | BF16 donanım | FP8 | YuE2 için durum |
|---|---|---|---|---|---|
| T4 | 16 GB | 7.5 | Hayır | Hayır | Sığıyor ama yavaş; `--offload-ar` ve düşük `--budget` gerekebilir |
| RTX 3090 | 24 GB | 8.6 | Evet | Hayır | Resmi 24 GB hedefiyle uyumlu |
| A100 | 40/80 GB | 8.0 | Evet | Hayır | Bol bellek, yüksek bant genişliği |
| **L4** | 24 GB | 8.9 | Evet | Evet | Resmi hedefle birebir uyumlu, Colab için önerilen |
| RTX 4090 | 24 GB | 8.9 | Evet | Evet | Masaüstünde en uygun seçenek |
| H100 | 80 GB | 9.0 | Evet | Evet | En hızlı, en pahalı |

Modelin bellek ihtiyacı (`config.json`'dan hesaplanmış yaklaşık değerler):

| Kalem | Boyut |
|---|---|
| Ağırlıklar (BF16) | 7.3 GB |
| KV cache | token başına ~112 KB (28 katman × 8 KV kafası × 128 boyut × 2 × 2 bayt); 9.000 token ~1 GB, tam bağlam ~2.8 GB |
| CFG ikinci dalı | Kullanılırsa KV cache iki katına çıkar |
| Sentez + VAE | ~1-3 GB |
| Pipeline'ın ayırdığı pay | 2 GB |
| **Toplam** | **~12-17 GB**; resmi 24 GB hedefi uzun şarkılar ve CFG için pay bırakıyor |

### Dürüst bir not

Bu kurulum yalnızca Mac M4 (MPS) üzerinde ölçüldü. NVIDIA tarafındaki avantajlar yue2'nin kodundan ve
NVIDIA'nın donanım özelliklerinden çıkarıldı; **L4 ya da başka bir NVIDIA kartta hız henüz ölçülmedi.** Colab
paketi CUDA için hazırlandı ama çalıştırılmadı. FP8 ve 16 adımlı hızlı sentezin ses kalitesine etkisi de
karşılaştırmalı dinlenmedi.
