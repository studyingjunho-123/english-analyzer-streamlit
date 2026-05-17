import streamlit as st
from openai import OpenAI
import base64

# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="내신 영어 지문 분석기",
    page_icon="📘",
    layout="wide"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# =========================
# 프롬프트
# =========================

OCR_PROMPT = """
너는 영어 지문 OCR 보조자다.

이미지 속 영어 지문을 최대한 정확히 추출해라.

규칙:
- 이미지에 보이는 영어 지문 본문만 원문 그대로 옮겨라.
- 철자, 대소문자, 문장부호를 최대한 유지해라.
- 줄바꿈은 자연스럽게 정리하되, 문장 순서는 바꾸지 마라.
- 보이지 않거나 확실하지 않은 단어는 추측하지 말고 [확인 필요]라고 표시해라.
- 문제 번호, 선택지, 필기 흔적, 페이지 번호는 지문 분석에 필요하지 않으면 제외해라.
- 한국어 설명을 덧붙이지 말고, 추출된 영어 지문만 출력해라.
"""

ANALYSIS_PROMPT = """
너는 한국 고등학교 영어 내신 지문 분석 전문가다.

분석 원칙:
- 반드시 제공된 영어 지문 안의 내용만 근거로 분석한다.
- 지문에 없는 배경지식이나 추측을 넣지 않는다.
- 확실하지 않은 부분은 단정하지 말고 "확인 필요"라고 표시한다.
- 해석은 문장 단위로 제공한다.
- 문법 설명은 원문 표현을 직접 근거로 설명한다.
- 학생이 내신 시험 전에 이해하고 암기할 수 있도록 구체적으로 분석한다.
- 너무 어려운 문법 용어만 나열하지 말고, 고등학생이 이해할 수 있게 설명한다.
- 지문 내용을 과하게 확대 해석하지 않는다.

분석할 영어 지문:
{passage}

출력 형식:

# 1. 전체 주제
- 영어 주제문:
- 한국어 주제:
- 핵심 한 줄 요약:

# 2. 핵심 키워드
중요 키워드 5~8개를 정리해라.
각 키워드마다 지문 속 의미를 함께 설명해라.

# 3. 글의 흐름 구조
글의 전개를 다음 형식으로 정리해라.

- 도입:
- 전개:
- 전환/대조:
- 결론:

그리고 아래처럼 화살표 구조로도 정리해라.

예:
문제 제기 → 원인 설명 → 예시 제시 → 결론

# 4. 문장별 분석
각 문장을 번호로 나누어 분석해라.

각 문장마다 반드시 다음 형식을 지켜라.

[문장 1]
원문:
직역:
자연스러운 해석:
핵심 문법:
중요 표현:
내용상 역할:
내신 포인트:

[문장 2]
원문:
직역:
자연스러운 해석:
핵심 문법:
중요 표현:
내용상 역할:
내신 포인트:

# 5. 중요한 문법 포인트
지문에 실제로 등장하는 문법만 골라 설명해라.

가능한 항목:
- 관계사
- 분사 / 분사구문
- 수동태
- 가정법
- 비교
- 병렬구조
- 접속사 / 전환어
- 도치
- 강조
- 대명사 지칭
- 어법상 주의할 부분

지문에 없는 문법은 억지로 만들지 마라.

# 6. 중요한 내용 포인트
내신 시험에서 내용 이해 문제로 나올 수 있는 부분을 정리해라.

- 글쓴이의 주장:
- 핵심 대조:
- 원인과 결과:
- 예시의 역할:
- 마지막 문장의 의미:
- 헷갈리기 쉬운 내용:

# 7. 나올 만한 문제 유형
다음 유형별로 출제 가능성을 분석해라.

1) 주제 / 제목:
2) 빈칸 추론:
3) 문장 삽입:
4) 순서 배열:
5) 어법:
6) 어휘:
7) 내용 일치 / 불일치:
8) 요약문 완성:
9) 문맥상 낱말 바꾸기:

각 항목마다 근거가 되는 원문 표현을 함께 제시해라.

# 8. 함정 포인트
학생이 헷갈릴 만한 부분을 정리해라.

- 해석상 함정:
- 문법상 함정:
- 내용상 함정:
- 선택지로 꼬기 쉬운 부분:

# 9. 주제문 후보
지문에서 주제문 역할을 할 수 있는 문장을 원문 그대로 1~2개 골라라.
각 문장이 왜 중요한지 설명해라.

# 10. 시험 직전 암기 정리
시험 직전에 볼 수 있게 압축 정리해라.

- 핵심 내용:
- 핵심 문법:
- 핵심 어휘:
- 가장 조심할 함정:
"""


# =========================
# 함수
# =========================

def image_to_base64(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = uploaded_file.type
    return f"data:{mime_type};base64,{encoded}"


def extract_text_from_image(uploaded_file):
    image_data_url = image_to_base64(uploaded_file)

    response = client.responses.create(
        model="gpt-5.5",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": OCR_PROMPT
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url
                    }
                ]
            }
        ]
    )

    return response.output_text


def analyze_passage(passage):
    prompt = ANALYSIS_PROMPT.format(passage=passage)

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt
    )

    return response.output_text


# =========================
# UI
# =========================

st.title("📘 내신 영어 지문 분석기")
st.caption("영어 지문을 사진 또는 텍스트로 입력하면 문장별 해석, 문법, 주제, 출제 포인트를 분석합니다.")

st.info("""
📌 더 정확한 분석을 위해 태블릿 캡처본처럼 글자가 선명하고 기울어지지 않은 이미지를 업로드해 주세요.

사진이 흐리거나 글자가 작거나 필기/그림자가 많으면 지문 인식이 부정확할 수 있습니다.

AI의 분석에는 오류가 있을 수 있으므로, 중요한 내용은 반드시 원문과 교과서/수업 자료를 기준으로 한 번 더 확인해 주세요.
""")

tab1, tab2 = st.tabs(["📷 사진으로 분석", "⌨️ 텍스트로 분석"])


# =========================
# 사진 분석 탭
# =========================

with tab1:
    st.subheader("📷 지문 사진 업로드")

    uploaded_file = st.file_uploader(
        "영어 지문 사진을 업로드하세요.",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="업로드한 이미지", use_container_width=True)

        if st.button("1단계: 이미지에서 지문 추출"):
            with st.spinner("이미지에서 영어 지문을 추출하는 중..."):
                extracted_text = extract_text_from_image(uploaded_file)
                st.session_state["extracted_text"] = extracted_text
                st.session_state.pop("analysis_result", None)

        if "extracted_text" in st.session_state:
            st.subheader("✏️ 추출된 지문 확인 및 수정")
            corrected_text = st.text_area(
                "AI가 추출한 지문입니다. 오타나 누락이 있으면 수정하세요.",
                value=st.session_state["extracted_text"],
                height=300
            )

            if st.button("2단계: 내신용 지문 분석"):
                with st.spinner("내신용 분석을 생성하는 중..."):
                    result = analyze_passage(corrected_text)
                    st.session_state["analysis_result"] = result

            if "analysis_result" in st.session_state:
                st.subheader("📌 분석 결과")
                st.markdown(st.session_state["analysis_result"])


# =========================
# 텍스트 분석 탭
# =========================

with tab2:
    st.subheader("⌨️ 영어 지문 직접 입력")

    passage = st.text_area(
        "영어 지문을 붙여넣으세요.",
        height=300,
        placeholder="여기에 영어 지문을 입력하세요."
    )

    if st.button("내신용 지문 분석하기"):
        if passage.strip() == "":
            st.warning("분석할 영어 지문을 입력해 주세요.")
        else:
            with st.spinner("내신용 분석을 생성하는 중..."):
                result = analyze_passage(passage)
                st.subheader("📌 분석 결과")
                st.markdown(result)