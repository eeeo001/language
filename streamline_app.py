import streamlit as st
import collections
import math
import pandas as pd
from typing import List, Dict, Any, Tuple

# --- 1. 영어 규칙 (비교 기준) 정의 ---
ENGLISH_RULES = {
    "vowels": 15,            # 약 15개 (단모음, 이중모음 포함)
    "consonants": 24,        # 약 24개
    "word_order": "SVO",     # 주어-동사-목적어
}

# --- 2. 평가 코어 함수 (이전과 동일) ---

@st.cache_data
def calculate_entropy(characters: List[str]) -> float:
    """글자 기반 조건부 엔트로피를 계산합니다. (기존과 동일)"""
    if len(characters) < 2: return 0.0
    
    bigrams = collections.defaultdict(int)
    unigrams = collections.defaultdict(int)
    
    for i in range(len(characters) - 1):
        bigram = (characters[i], characters[i+1])
        bigrams[bigram] += 1
        unigrams[characters[i]] += 1
    unigrams[characters[-1]] += 1

    conditional_entropy = 0
    for (char1, char2), count in bigrams.items():
        if unigrams[char1] > 0:
            p_char1 = unigrams[char1] / len(characters)
            p_char2_given_char1 = count / unigrams[char1]
            
            if p_char1 > 0 and p_char2_given_char1 > 0:
                conditional_entropy += (p_char1 * p_char2_given_char1) * math.log2(p_char2_given_char1)
                
    return abs(conditional_entropy)


@st.cache_data
def evaluate_language(corpus: List[Dict[str, str]], rules: Dict[str, Any], language_key: str) -> Dict[str, float]:
    """예외 항목을 제외하고 효율성 지표를 계산합니다. (기존과 동일)"""
    
    language_text = " ".join([d[language_key] for d in corpus])
    meaning_text = " ".join([d.get("meaning_english") or d.get("english") for d in corpus])
        
    language_chars = list(language_text.replace(" ", ""))
    language_words = language_text.split()
    unique_language_words = set(language_words)
    
    total_meaning_tokens = len(meaning_text.split())
    total_language_chars = len(language_chars)
    semantic_density_per_char = total_meaning_tokens / total_language_chars if total_language_chars > 0 else 0
    
    final_entropy = calculate_entropy(language_chars)
    word_regularity = len(unique_language_words) / len(language_words) if len(language_words) > 0 else 0
    
    # 복잡성 지수 = 음소 목록 크기 * 0.2
    phoneme_inventory_size = rules["vowels"] + rules["consonants"]
    morph_complexity = phoneme_inventory_size * 0.2
    
    return {
        "의미_밀도_지수": semantic_density_per_char,
        "엔트로피_지수": final_entropy,
        "규칙성_지수": word_regularity,
        "복잡성_지수": morph_complexity,
        "문장_구조": rules["word_order"]
    }

# --- 3. 데이터 처리 함수 (새로 작성됨) ---

@st.cache_data
def parse_conlang_data(raw_conlang: str, raw_english: str) -> Tuple[List[Dict[str, str]], str]:
    """나란히 입력된 텍스트를 코퍼스 리스트로 파싱하고 유효성 검사를 수행합니다."""
    
    conlang_lines = [line.strip() for line in raw_conlang.strip().split('\n') if line.strip()]
    english_lines = [line.strip() for line in raw_english.strip().split('\n') if line.strip()]

    if not conlang_lines or not english_lines:
        return [], "데이터가 비어 있습니다."
        
    if len(conlang_lines) != len(english_lines):
        return [], f"인공언어 문장 수 ({len(conlang_lines)}개)와 영어 번역 문장 수 ({len(english_lines)}개)가 일치하지 않습니다."

    corpus = []
    for c_sent, e_sent in zip(conlang_lines, english_lines):
        corpus.append({"conlang": c_sent, "meaning_english": e_sent})
            
    return corpus, ""

# --- 4. Streamlit 앱 본체 ---

def app():
    st.set_page_config(layout="wide")
    st.title("인공언어 vs. 영어 효율성 비교 분석 (음운/통사 중심)")
    st.markdown("---")

    # --- A. 사용자 입력 섹션 ---
    col_input, col_rules = st.columns([2, 1])

    with col_input:
        st.subheader("1. 인공언어 코퍼스 입력")
        
        # 나란히 입력할 두 개의 열 생성
        col_conlang, col_english = st.columns(2)
        
        initial_conlang = (
            "alo kata cemi\ncemi alo zemi\nkata alo\ntelo mi\ntelo kato"
        )
        initial_english = (
            "I see a good house\nThe house helps me\nThe good I\nThere is water\nThere is no water"
        )
        
        raw_conlang = col_conlang.text_area(
            "인공언어 문장 (한 줄에 하나의 문장)",
            value=initial_conlang,
            height=250,
            help="각 줄은 오른쪽의 영어 번역과 1:1로 대응됩니다."
        )
        
        raw_english = col_english.text_area(
            "영어 번역 (한 줄에 하나의 문장)",
            value=initial_english,
            height=250
        )

    with col_rules:
        st.subheader("2. 규칙 및 복잡성 정의")
        
        # 인공언어 규칙 입력
        st.markdown("##### 인공언어 (CONLANG)")
        c_vowels = st.number_input("모음 개수 (Vowels)", min_value=1, max_value=20, value=5, key='c_vowels')
        c_consonants = st.number_input("자음 개수 (Consonants)", min_value=1, max_value=40, value=15, key='c_consonants')
        c_word_order = st.selectbox("문장 구조 (Word Order)", options=['SVO (주어-동사-목적어)', 'SOV (주어-목적어-동사)', '기타'], index=0, key='c_order')

        st.markdown("##### 영어 규칙 (비교 기준)")
        st.caption(f"영어 기준: Vowels {ENGLISH_RULES['vowels']}, Consonants {ENGLISH_RULES['consonants']}, Order {ENGLISH_RULES['word_order']}")


    # 데이터 파싱 및 유효성 검사
    conlang_corpus, error_msg = parse_conlang_data(raw_conlang, raw_english)
    if error_msg:
        st.error(f"오류: {error_msg}")
        return
    if not conlang_corpus:
        st.warning("경고: 코퍼스 데이터가 비어 있습니다.")
        return

    # --- B. 데이터 준비 및 평가 ---
    
    CONLANG_RULES = {
        "vowels": c_vowels,
        "consonants": c_consonants,
        "word_order": c_word_order.split(' ')[0], 
    }
    
    ENGLISH_DATA = {
        "corpus": [{"english": d["meaning_english"]} for d in conlang_corpus],
        "rules": ENGLISH_RULES 
    }
    
    # 평가 실행
    conlang_results = evaluate_language(conlang_corpus, CONLANG_RULES, language_key="conlang")
    english_results = evaluate_language(ENGLISH_DATA["corpus"], ENGLISH_RULES, language_key="english") 

    # --- C. 결과 출력 섹션 ---
    
    st.markdown("---")
    st.header("인공언어 vs. 영어 효율성 비교 결과")

    results_data = {
        "평가 항목": [
            "1. 글자당 의미 밀도 (압축성)", 
            "2. 조건부 엔트로피 (예측성)", 
            "3. 음운/통사 복잡성 지수", 
            "4. 어휘적 규칙성 지수"
        ],
        "인공언어 점수": [
            conlang_results["의미_밀도_지수"], 
            conlang_results["엔트로피_지수"], 
            conlang_results["복잡성_지수"], 
            conlang_results["규칙성_지수"]
        ],
        "영어 점수": [
            english_results["의미_밀도_지수"], 
            english_results["엔트로피_지수"], 
            english_results["복잡성_지수"], 
            english_results["규칙성_지수"]
        ],
        "높은 값이 효율적인가?": [True, False, False, True]
    }
    
    results_df = pd.DataFrame(results_data)
    
    # 비교 결과 (⭐ 표시)
    def get_comparison_star(row):
        is_conlang_better = (row['인공언어 점수'] > row['영어 점수']) if row['높은 값이 효율적인가?'] else (row['인공언어 점수'] < row['영어 점수'])
        
        if abs(row['인공언어 점수'] - row['영어 점수']) < 0.0001: 
             return "DRAW"
        return "⭐" if is_conlang_better else ""

    results_df['인공언어 효율'] = results_df.apply(lambda row: get_comparison_star(row), axis=1)
    
    # 최종 테이블 출력
    st.dataframe(
        results_df[['평가 항목', '인공언어 점수', '인공언어 효율', '영어 점수']],
        column_config={
            "인공언어 점수": st.column_config.NumberColumn(format="%.4f"),
            "영어 점수": st.column_config.NumberColumn(format="%.4f"),
            "인공언어 효율": st.column_config.TextColumn("CONLANG (WIN/DRAW)")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.subheader("추가 정보")
    st.write(f"**인공언어 문장 구조:** {CONLANG_RULES['word_order']} | **영어 문장 구조:** {ENGLISH_RULES['word_order']}")
    st.write(f"**인공언어 음소 목록 크기 (V+C):** {CONLANG_RULES['vowels'] + CONLANG_RULES['consonants']} | **영어 음소 목록 크기 (V+C):** {ENGLISH_RULES['vowels'] + ENGLISH_RULES['consonants']}")
    
    st.markdown("""
        ---
        **해석:**
        - **압축성 (높을수록 효율적):** 메시지를 전달하는 데 필요한 글자 수가 적습니다.
        - **엔트로피 (낮을수록 효율적):** 언어 구조가 더 예측 가능하여 중복성이 높습니다.
        - **음운/통사 복잡성 (낮을수록 효율적):** 음소 목록 크기(V+C)만을 기준으로 한 학습 난이도입니다.
        - **규칙성 (높을수록 효율적):** 어휘 학습 시 불규칙성이 적습니다.
    """)


if __name__ == "__main__":
    
    # ENGLISH_DATA는 conlang_corpus의 영어 번역을 사용하기 위해 빈 구조로 초기화
    ENGLISH_DATA = {"corpus": [], "rules": {}}
    
    app()
