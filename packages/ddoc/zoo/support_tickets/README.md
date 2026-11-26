# support_tickets — 앱 업데이트 뒤의 고객 문의 (텍스트)

메모 앱에 들어오는 고객 문의를 **정답이 있게 만든** 텍스트 사례입니다. 평범한 한 주는 요금, 사용법, 오류 신고,
계정 변경, 로그인 문제, 감사 인사가 섞여 있고, 문의마다 인사말·맺음말·표현이 조금씩 다릅니다. 이야기마다 한
가지만 바꿉니다. 판정은 text 플러그인이 맡습니다(`./run.sh setup text`, torch 필요).

```bash
python make.py                    # 앱 업데이트 (기본) → data/ref/, data/cur/
ddoc diff data/ref data/cur --with-embeddings --lang ko --report note.html
ddoc examples tour support        # 아래 이야기 셋을 한 단계씩 (투어)
```

데이터는 [`ddoc.examples.support`](../../ddoc/examples/support.py) 가 고정 시드로 만듭니다(쪽마다 200건). 폴더마다
`tickets/data.csv`(id, text)와, 어느 열이 글인지 적은 `ddoc.yaml`(`modality: text`, `text_column: text`)이
있습니다. 글은 영어입니다. text 플러그인의 지표(불용어 비율, 가독성)와 임베딩이 영어 기준이기 때문입니다.

## 이야기와 결과

`--with-embeddings` 로 비교한 결과입니다. **이웃 섞임**은 1이면 고르게 섞였고 0이면 완전히 갈라졌습니다.

| story | 현재 | 판정 (점수) | 가장 변한 속성 | 이웃 섞임 | 판독 |
|---|---|---|---|---|---|
| `app-update` | 버전 5 출시 다음 주 | **DRIFT** (0.268) | 글자 수 0.529 · 단어 수 0.506 · 어휘 다양성 0.443 | 0.63 | 레짐 전환형 · 길이 이동·확산형 |
| `chat-widget` | 같은 주제, 채팅 창 | **DRIFT** (0.504) | 단어 수 0.628 · 특수문자 0.613 · 임베딩 분리 0.609 | 0.39 | 임베딩 분리형 · 길이 이동·수축형 · 불용어 수준 이동형 |
| `two-weeks` | 평범한 다음 주 | OK (0.000) | — | 1.00 | 안정형 |

점수는 속성마다 JS 거리(표본 잡음으로 설명되면 0)이고, 임베딩 지도는 `embedding_separation`(임베딩 분리 =
1 − 이웃 섞임)으로 한 속성이 됩니다.

**앱 업데이트 (`app-update`).** 로그인 문의가 12%에서 34%로 늘고, 없던 동기화 불만이 22% 생겼으며, 불만 문의에
"벌써 두 번 재설치했다", "업무에 쓰는데 빨리 고쳐 달라" 같은 문장이 붙어 평균 단어 수가 19에서 30으로
늘었습니다. 주제는 일부만 바뀌어(요금·사용법 문의는 그대로 있음) 지도에서 두 주가 꽤 섞이지만(0.63), 진단서가
꼽은 "기준과 가장 다른 현재 문의" 다섯 건 중 네 건이 버전 5 이야기입니다. 지도 아래의 항목 이름
(`tickets/data.csv/33` = 33번 행)으로 원문을 열어 보세요.

**채팅 위젯 (`chat-widget`).** 새로 묻는 것은 없고 쓰는 방식만 바뀌었습니다. 인사 없이 짧게, 소문자로,
"pls"·"cant" 같은 줄임말로 씁니다(평균 단어 수 19 → 12). 그런데 지도에서는 앱 업데이트보다 **더 갈라집니다**
(0.39). 임베딩은 무슨 말을 하는지뿐 아니라 어떻게 쓰는지도 담습니다. 이메일로 학습한 분류 모델은 요청이 같아도
채팅 문장에서 헤맬 수 있다는 뜻입니다. 길이는 짧아지면서 고르게 모여 이동·수축형으로 읽힙니다.

**평범한 두 주 (`two-weeks`).** 같은 규칙으로 쓴 다음 주입니다. OK, 판독은 안정형, 지도는 완전히 섞입니다(1.00).

## 내 텍스트로

```text
before/tickets/data.csv     id,text 가 있는 CSV (열 이름은 자유)
before/tickets/ddoc.yaml    modality: text
                            text_column: text
                            id_column: id
after/…                     같은 구조
```

```bash
ddoc diff before after --with-embeddings --report note.html
```
