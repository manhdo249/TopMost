from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy as np
from tqdm import tqdm
from ..data.file_utils import split_text_word


def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    """
    Tính toán độ nhất quán chủ đề cho một bộ văn bản tham chiếu dựa trên từ vựng và các từ chủ đề.

    Tham số:
        - reference_corpus: Bộ văn bản tham chiếu để tính độ nhất quán.
        - vocab: Từ vựng (tập hợp các từ) được sử dụng trong bộ văn bản.
        - top_words: Danh sách các từ đứng đầu trong mỗi chủ đề.
        - cv_type: Loại độ nhất quán cần tính (mặc định là 'c_v').

    Trả về:
        - score: Điểm độ nhất quán trung bình của các chủ đề.
    """
    # Tách từ trong danh sách từ chủ đề
    split_top_words = split_text_word(top_words)
    num_top_words = len(split_top_words[0])
    for item in split_top_words:
        assert num_top_words == len(item) # Kiểm tra rằng tất cả các danh sách từ có cùng số lượng từ

    # Tách từ trong bộ văn bản tham chiếu
    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab)) # Tạo từ điển từ tập từ vựng
    
    # Tạo mô hình độ nhất quán chủ đề
    cm = CoherenceModel(texts=split_reference_corpus, dictionary=dictionary, topics=split_top_words, topn=num_top_words, coherence=cv_type)
    cv_per_topic = cm.get_coherence_per_topic() # Tính toán độ nhất quán cho từng chủ đề
    
    # Tính điểm độ nhất quán trung bình
    score = np.mean(cv_per_topic)
    return score


def dynamic_TC(train_texts, train_times, vocab, top_words_list, cv_type='c_v', verbose=False):
    """
    Tính toán độ nhất quán chủ đề động theo thời gian.

    Tham số:
        - train_texts: Danh sách các văn bản trong tập huấn luyện.
        - train_times: Danh sách các thời điểm tương ứng với mỗi văn bản trong train_texts.
        - vocab: Từ vựng được sử dụng.
        - top_words_list: Danh sách các từ đứng đầu cho mỗi chủ đề tại mỗi thời điểm.
        - cv_type: Loại độ nhất quán cần tính (mặc định là 'c_v').
        - verbose: Nếu là True, sẽ in ra danh sách điểm độ nhất quán theo thời gian.

    Trả về:
        - Điểm trung bình của tất cả các điểm độ nhất quán theo thời gian.
    """
    cv_score_list = list()

    for time, top_words in tqdm(enumerate(top_words_list)):
        # use the texts of each time slice as the reference corpus.
        idx = np.where(train_times == time)[0]
        reference_corpus = [train_texts[i] for i in idx]

        # use the topics at a time slice
        cv_score = compute_topic_coherence(reference_corpus, vocab, top_words, cv_type)
        cv_score_list.append(cv_score)

    if verbose:
        print(f"dynamic TC list: {cv_score_list}")

    return np.mean(cv_score_list)
