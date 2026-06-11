# Local AI Stack — llama.cpp (GGUF + MTP) + Open WebUI

GGUF 모델을 **MTP + 멀티 GPU**로 서빙하고, **Open WebUI**로 계정/대화를 관리하는 Docker 스택.

```
[Open WebUI :3000] ──OpenAI /v1──▶ [llm :8010→8000] ──llama.cpp──▶ GPU (멀티 GPU 텐서분할)
       계정·대화(volume)              GGUF + MTP(volume)
[ComfyUI :8188] (옵션: --profile comfyui)
```

- 백엔드 = `llama-server` 직접 구동: `--tensor-split`(전 GPU) + `--spec-type draft-mtp`(MTP), OpenAI `/v1`.
- `llm`은 `unsloth/unsloth` 이미지의 **prebuilt llama.cpp 바이너리만** 사용(MTP 지원 빌드). Studio는 안 띄움.
- 검증: Qwen3.6-27B-MTP UD-Q4_K_XL, 2×3090 → ~60 tok/s.

## 사전 준비

```bash
# NVIDIA 드라이버 + Container Toolkit 설치 후, nvidia 런타임 등록:
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # GPU 보이면 OK
```

## 모델 다운로드 (호스트에서)

```bash
cp .env.example .env
sudo apt install -y python3-venv
python3 -m venv .venv && .venv/bin/pip install -U "huggingface_hub[hf_transfer]"

.venv/bin/python download_model.py unsloth/Qwen3.6-27B-MTP-GGUF -i "*UD-Q4_K_XL*"
# gated 모델(gemma 등): HF_TOKEN=hf_xxx .venv/bin/python download_model.py <repo> -i "*UD-Q4_K_XL*" "mmproj*"
```
- `./volume/models/hub/...`(HF 캐시)에 저장 → 호스트에서 직접 관리·영구 보존.
- 스크립트가 출력하는 `MODEL_HUB`/`MODEL_GLOB`를 `.env`에 반영.
- "UD"(Unsloth Dynamic)도 표준 GGUF라 일반 llama.cpp로 동작.

## 기동

```bash
# 강한 세션 키 생성 (한 번만; 이후 고정해야 로그인 세션 유지)
sed -i "s|^WEBUI_SECRET_KEY=.*|WEBUI_SECRET_KEY=$(openssl rand -hex 32)|" .env
nano .env                          # 필요시 LLM_CTX / LLM_TENSOR_SPLIT / LLM_GPUS 조정

docker compose up -d llm open-webui
docker compose ps                  # 둘 다 healthy (llm은 모델 로드까지 수십 초~분)
```
→ `http://<host>:3000` 접속, 첫 가입 계정이 관리자. 모델은 `.env` 기준 자동 로드.

## 볼륨 (컨테이너 재생성해도 보존)

| 경로 | 내용 |
|---|---|
| `./volume/models/` | GGUF 모델 |
| `./volume/open-webui/` | 계정·설정·대화 |
| `./volume/comfyui/{models,output,user}/` | ComfyUI 데이터 |

## GPU 구성

`.env`의 `LLM_GPUS`(사용할 GPU) + `LLM_TENSOR_SPLIT`(분할 비율)로 제어. **둘의 GPU 개수를 일치**시킬 것.

**여러 장에 나눠 담기 (큰 모델 — 기본)** — 1장에 안 들어가는 모델용. 예: 27B Q4(가중치 17GB + KV + MTP > 24GB).
```
LLM_GPUS=all
LLM_TENSOR_SPLIT=1,1        # 2장. 4장이면 1,1,1,1
```

**1장에 들어가는 모델** — 예: ≤14GB급(gemma-12b 등) 또는 27B Q3(~13GB). 분할 sync가 없어 약간 더 빠르고, 나머지 GPU를 비워 다른 모델용으로 확보.
```
LLM_GPUS=0                 # GPU0만 사용 (GPU1은 비움)
LLM_TENSOR_SPLIT=1
LLM_CTX=16384              # 1장이 빠듯하면 컨텍스트 축소 (32k에서 추론버퍼 OOM 시)
```

> ⚠️ **토큰 생성 속도는 GPU 개수와 거의 무관**합니다. 레이어 분할은 GPU가 순차로 동작(한 번에 한 장)하므로, 단일 요청 속도 ≈ GPU 한 장 성능. 2장은 "속도"가 아니라 "용량(큰 모델/긴 컨텍스트)"용입니다. 더 빠른 생성을 원하면 **더 작은 모델**(12~14B는 27B 대비 ~2배)을 쓰세요.

변경 후: `docker compose up -d llm`. 확인: `docker exec llm nvidia-smi`

## 외부 클라이언트 (OpenClaw / Cursor / Continue / Cline 등)

OpenAI 호환 엔드포인트를 받는 도구는 모두 연결됩니다.

| 항목 | 값 |
|---|---|
| Base URL | `http://<host>:8010/v1` |
| API Key | `.env`의 `LLM_API_KEY` (기본 `sk-local`) |
| Model | `/v1/models`의 id (예: `Qwen3.6-27B-UD-Q4_K_XL.gguf`) |

```bash
curl -s http://<host>:8010/v1/models -H "Authorization: Bearer sk-local" | jq -r '.data[].id'
```
> 외부 노출 시 `LLM_API_KEY`를 강한 값으로 바꾸고 방화벽/TLS 뒤에 둘 것.

## 속도 측정 (tok/s)

```bash
curl -s http://localhost:8010/v1/chat/completions -H "Authorization: Bearer sk-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-27B-UD-Q4_K_XL.gguf","messages":[{"role":"user","content":"Count to 100."}],"max_tokens":300}' \
  | jq '{tok_s:.timings.predicted_per_second, draft_acc:(.timings.draft_n_accepted/.timings.draft_n)}'
```
> tok/s는 내용(MTP 수용률)에 따라 변동. 다중 사용자 동시 요청 시 슬롯 분할로 각자 느려짐(정상).

## 운영

```bash
docker compose logs -f llm
docker compose restart llm                     # 모델/.env 변경 반영
docker compose pull && docker compose up -d    # 이미지 업데이트
docker compose --profile comfyui up -d         # ComfyUI 추가 (:8188)
```

## 트러블슈팅

- **`could not select device driver "nvidia"`** → `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`
- **llm 바로 죽음 / "no GGUF found"** → 모델 미다운로드 또는 `MODEL_HUB`/`MODEL_GLOB` 불일치. `docker exec llm find /workspace/.cache/huggingface/hub -name "*.gguf"`
- **`CUDA error: out of memory`** → `LLM_CTX` 축소 또는 `LLM_TENSOR_SPLIT`로 GPU 더 분할. `docker compose restart llm`
- **OWUI에 모델 안 보임** → `docker compose logs llm`에 "server is listening" 확인. Base URL `http://llm:8000/v1`, 키 `LLM_API_KEY`.

## 참고

- llama.cpp: https://github.com/ggml-org/llama.cpp · Open WebUI: https://docs.openwebui.com/
