import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(filepath):
    data = pd.read_csv(filepath)
    questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
    answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
    return questions, answers    

    
def calc_distance(a, b):
    ''' 레벤슈타인 거리 계산하기 '''
    if a == b: return 0 # 같으면 0을 반환
    a_len = len(a) # a 길이
    b_len = len(b) # b 길이
    if a == "": return b_len
    if b == "": return a_len
   
    matrix = [[] for i in range(a_len+1)] # 리스트 컴프리헨션을 사용하여 1차원 초기화
    for i in range(a_len+1): # 0으로 초기화
        matrix[i] = [0 for j in range(b_len+1)]  # 리스트 컴프리헨션을 사용하여 2차원 초기화
    # 0일 때 초깃값을 설정
    for i in range(a_len+1):
        matrix[i][0] = i
    for j in range(b_len+1):
        matrix[0][j] = j
# 표 채우기 --- (※2) 2차원 배열로 채워짐
    
    for i in range(1, a_len+1):
        ac = a[i-1]
    # print(ac,'=============')
        for j in range(1, b_len+1):
            bc = b[j-1] 
        # print(bc)
            cost = 0 if (ac == bc) else 1  #  파이썬 조건 표현식 예:) result = value1 if condition else value2
            matrix[i][j] = min([
                matrix[i-1][j] + 1,     # 문자 제거: 위쪽에서 +1
                matrix[i][j-1] + 1,     # 문자 삽입: 왼쪽 수에서 +1   
                matrix[i-1][j-1] + cost # 문자 변경: 대각선에서 +1, 문자가 동일하면 대각선 숫자 복사
            ])
                # print(matrix)
        # print(matrix,'----------끝')
    return matrix[a_len][b_len]
    

    
def find_best_answer(input_sentense):
    # 레벤슈타인 거리 계산 후 거리가 가장 작은 값 리스트 추출
    
    le_questions = []
    le_answers = []    
        
    index = 0
    for str in samples[0]:
        if index == 0:
            n= calc_distance(input_sentense,str) #레벤슈타인 거리 기억
            if len(le_questions) > 0:
                del le_questions[0:]
                del le_answers[0:]
            le_questions.append(str) #레벤슈타인 거리계산으로 나온 질문 리스트에 추가
            le_answers.append(samples[1][index])
            index += 1
        else:
            #기억 하고 있는 레벤슈타인 거리와 현재 계산 된 레벤슈타인 거리 비교
            if n == calc_distance(input_sentense,str): #비교한 결과가 같으면 리스트에 데이터 추가
                le_questions.append(str)
                le_answers.append(samples[1][index])
            elif n > calc_distance(input_sentense,str): #현재 나온 레벤슈타인 거리가 작으면 리스트에 데이터 삭제 후 추가
                n= calc_distance(input_sentense,str)
                del le_questions[0:]
                del le_answers[0:]
                le_questions.append(str)
                le_answers.append(samples[1][index])
            index += 1
        
    if len(le_questions) == 1: #레벤슈타인 거리로 추출한 질문 리스트 데이터 개수가 1개면 바로 답변으로 채택
        return le_answers[0]
    else: #레벤슈타인 거리로 추출한 질문 리스트 데이터 개수가 1개 이상일 경우 주출하여 나온 리스트를 다시 기존 TF-IDF와 Consin Similarlity 를 이용하여 답변 채택
        
        question_vectors = vectorizer.fit_transform(le_questions)  # 질문을 TF-IDF로 변환
        input_vector = vectorizer.transform([input_sentence])
        similarities = cosine_similarity(input_vector, question_vectors) # 코사인 유사도 값들을 저장
        
        best_match_index = similarities.argmax()   # 유사도 값이 가장 큰 값의 인덱스를 반환
        return le_answers[best_match_index]  

filepath = 'T2.csv'
samples= load_data(filepath)

vectorizer = TfidfVectorizer()


while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = find_best_answer(input_sentence)
    print('Chatbot:', response)