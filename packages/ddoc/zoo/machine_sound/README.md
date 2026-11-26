# machine_sound — 소리가 달라지기 시작한 모터 (오디오)

전기 모터 소리를 **정답이 있게 만든** 오디오 사례입니다. 클립마다 1초(16 kHz)이고, 전원 험(120 Hz와 그
배음)·회전음(약 29.5 Hz)·바닥 잡음이 부하에 따라 조금씩 다릅니다. 판정은 audio 플러그인이 맡습니다
(`./run.sh setup audio`, librosa).

```bash
python make.py                    # 베어링 마모 (기본) → data/ref/, data/cur/
ddoc diff data/ref data/cur --lang ko --report note.html
ddoc examples tour machine        # 아래 이야기 셋을 한 단계씩 (투어)
```

데이터는 [`ddoc.examples.machine`](../../ddoc/examples/machine.py) 가 고정 시드로 만들어 표준 라이브러리로 WAV 를
씁니다(쪽마다 40개). 폴더마다 `machine/clip_*.wav` 와 `ddoc.yaml`(`modality: audio`)이 있습니다.

## 이야기와 결과

audio 플러그인은 클립마다 크기(RMS 에너지), 영교차율, 스펙트럼 중심을 잽니다.

| story | 현재 | 판정 (점수) | 평균 (기준 → 현재) | 판독 |
|---|---|---|---|---|
| `bearing-wear` | 베어링 결함 충격음 | **DRIFT** (80.76) | 스펙트럼 중심 2614 → 2856 Hz · 영교차율 0.061 → 0.113 | 스펙트럼 중심 이동·수축형 · 영교차율 이동·확산형 · 크기 수준 이동형 |
| `louder-mic` | 같은 소리, 2.2배 크게 | DRIFT (2.93) | 크기 0.161 → 0.354 · 나머지 그대로 | **크기 단위·스케일 변경형** |
| `two-shifts` | 같은 모터의 다른 클립 | DRIFT (2.87) | 모두 그대로 | 없음 |

**베어링 마모 (`bearing-wear`).** 결함 주파수(회전의 약 3.6배)마다 충격음이 나고, 그때마다 하우징이 3.2 kHz
근처에서 짧게 울립니다. 모터가 내지 않던 고주파가 생겨 스펙트럼 중심과 영교차율이 올라갑니다. 진단서에 값마다
기준과 현재의 분포 그래프가 있습니다.

**마이크를 가까이 (`louder-mic`).** 기계는 그대로이고 녹음이 2.2배 커졌습니다. 크기와 그 퍼짐이 같은 배율로
커지고 음높이 쪽 값은 그대로라, 판독이 **단위·스케일 변경형**으로 읽습니다. 기계보다 마이크의 위치와 이득을 먼저
보라는 뜻입니다.

## 실제로 알아 둘 점

**오디오 점수는 판정 기준의 척도가 아닙니다.** 대조군(`two-shifts`)은 바뀐 것이 없는데 DRIFT(2.87)이고,
마이크 이득 변경(2.93)과 점수로는 거의 구별되지 않습니다. audio 플러그인의 점수가 단위가 다른 값(Hz, 진폭)의
Wasserstein 거리를 그대로 평균해, 몇 Hz 의 우연한 차이도 기준값 0.15 를 넘기 때문입니다. 판독은 셋을 정확히
가릅니다(마모 · 크기만 바뀜 · 해당 없음). 오디오는 지금 **판정보다 판독을** 보세요. 플러그인 점수 척도를 맞추는
일은 [지금 상태](../../docs/status.md)의 알려진 한계에 있습니다.

## 내 녹음으로

```bash
mkdir -p before/clips after/clips
ffmpeg -i before.wav -f segment -segment_time 1 before/clips/%03d.wav
ffmpeg -i after.wav  -f segment -segment_time 1 after/clips/%03d.wav
printf 'modality: audio\nformats: [.wav]\n' | tee before/clips/ddoc.yaml > after/clips/ddoc.yaml
ddoc diff before after --report note.html
```
