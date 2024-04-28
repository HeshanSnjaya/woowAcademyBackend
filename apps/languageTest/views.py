import json
import re
import requests

import nltk
from django.db import transaction
from django.db.models import Sum
from django.http import JsonResponse
from .models import LanguageTest, Question, Answer, LanguageTestQuestion, QuestionAnswer, StudentLanguageTest
from django.shortcuts import get_object_or_404

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import numpy as np

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from rest_framework.decorators import api_view
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..users.models import Student

loaded_model = load_model('D:\DjangoProjects\woow-academy-backend\my_model.h5')
data = pd.read_csv("D:\DjangoProjects\woow-academy-backend\ielts_writing_dataset.csv")
X = data['essay']
y = data['marks']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
maxlen = len(max(X_seq, key=len))

# Load the keywords from the JSON file
with open('keywords.json', 'r') as f:
    keywords_data = json.load(f)

# Load JSON file
with open('D:/DjangoProjects/woow-academy-backend/answer.json', 'r') as file:
    sample_answers = json.load(file)

nltk.download('punkt')
nltk.download('stopwords')


# Endpoint to create a language test with random questions
@api_view(['GET'])
def get_random_questions(request):
    if request.method == 'GET':
        student_id = request.GET.get('studentId')
        if student_id:
            try:
                student = Student.objects.get(studentID=student_id)
            except Student.DoesNotExist:
                return JsonResponse({'error': 'Student not found.'}, status=404)

            total_questions = Question.objects.count()
            if total_questions < 10:
                return JsonResponse({'error': 'Not enough questions available.'}, status=400)

            with transaction.atomic():
                language_test = LanguageTest.objects.create()

                student_language_test = StudentLanguageTest.objects.create(
                    studentID=student,
                    languageTestId=language_test
                )

                all_question_ids = list(Question.objects.values_list('questionId', flat=True))
                random.shuffle(all_question_ids)
                random_question_ids = all_question_ids[:10]
                random_questions = Question.objects.filter(questionId__in=random_question_ids)

                for question in random_questions:
                    LanguageTestQuestion.objects.create(languageTestId=language_test, questionId=question)

            serialized_questions = [{
                'questionId': question.questionId,
                'questionDesc': question.questionDesc
            } for question in random_questions]

            response_data = {
                'languageTestId': language_test.languageTestId,
                'questions': serialized_questions
            }

            return JsonResponse(response_data)

        else:
            return JsonResponse({'error': 'Missing studentId parameter.'}, status=400)

    else:
        return JsonResponse({'error': 'Only GET requests are allowed.'}, status=405)


# Endpoint to submit an answer to a question in the language test
@api_view(['POST'])
def submit_answer(request):
    if request.method == 'POST':
        try:
            response = json.loads(request.body.decode('utf-8'))
            question_id = response.get('questionId')
            language_test_id = response.get('languageTestId')

            question = Question.objects.get(questionId=question_id)
            language_test = LanguageTest.objects.get(languageTestId=language_test_id)
            duration = float(response.get('duration', 0))
            answer_description = response.get('answerDesc')

            # keywords marks
            keywords_marks, keywords_list = assign_marks_from_keywords(question_id, answer_description)
            print("Keywords marks :", keywords_marks)

            # Keywords Order
            lower_keywords_list = [word.lower() for word in keywords_list]
            matching_keywords_from_answer = keyword_matching(answer_description.lower(), lower_keywords_list)
            matching_keywords_list = remove_different_keywords(lower_keywords_list, matching_keywords_from_answer)
            matching_words_count = count_words_in_correct_order(matching_keywords_list, matching_keywords_from_answer)
            keyword_order_score = (matching_words_count / len(matching_keywords_list)) * 10
            print("Keyword_order Score", keyword_order_score)

            # Grammar Check
            error_count = grammar_check(answer_description)
            print('error_count :', error_count)
            no_words_answer = len(answer_description.split())
            print('no_words_answer :', no_words_answer)
            grammar_score = ((no_words_answer - error_count) / no_words_answer) * 10
            print('grammar_score % :', grammar_score)

            # Predict Score from model
            rounded_score = predict_score(answer_description)

            # calculate the similarity
            filtered_sample_answers = [sample for sample in sample_answers if
                                       int(sample['marks']) == rounded_score and int(sample[
                                                                                         'questionId']) == int(
                                           question_id)]

            max_similarity = get_max_similarity(filtered_sample_answers, answer_description)
            print("Maximum Cosine Similarity Score:", max_similarity)

            additional_points = 0
            if 6 <= rounded_score <= 9:
                if duration <= 6:
                    additional_points = 0.5
                elif 6 < duration <= 8:
                    additional_points = 0.25

            model_total_score = rounded_score + additional_points
            print('Model Score : ' + str(rounded_score))
            print('Additional Time Score : ' + str(additional_points))
            print('model_total_score : ' + str(model_total_score))

            final_marks = ((model_total_score * 3) + (keywords_marks * 2.5) + (max_similarity * 2) + (grammar_score * 1.5) + keyword_order_score) / 10
            print('final_marks : ' + str(final_marks))

            with transaction.atomic():
                answer = Answer.objects.create(
                    answerDesc=answer_description,
                    marks=final_marks,
                    additionalPoints=additional_points
                )
                QuestionAnswer.objects.create(
                    questionId=question,
                    languageTestId=language_test,
                    answerId=answer,
                    duration=duration
                )
            return JsonResponse({'message': 'Answer saved successfully'}, status=201)

        except (Question.DoesNotExist, LanguageTest.DoesNotExist):
            return JsonResponse({'error': 'Invalid question ID or LanguageTest ID'}, status=400)

    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


def grammar_check(answer):
    url = "https://grammarbot.p.rapidapi.com/check"
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "X-RapidAPI-Key": "1f1c14c932mshd86ffe8e05de6dap1a9507jsnb7e7aa444443",
        "X-RapidAPI-Host": "grammarbot.p.rapidapi.com"
    }
    params = {
        "text": answer,
        "language": "en-US"
    }

    response = requests.post(url, headers=headers, params=params)

    if response.status_code == 200:
        json_data = response.json()
        matches_count = len(json_data.get("matches", []))
        return matches_count
    else:
        # Handle error cases
        return None


def cosine_similarity(X, Y):
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

    sw = stopwords.words('english')
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    rvector = X_set.union(Y_set)
    l1 = [1 if w in X_set else 0 for w in rvector]
    l2 = [1 if w in Y_set else 0 for w in rvector]

    dot_product = np.dot(l1, l2)
    magnitude_X = np.linalg.norm(l1)
    magnitude_Y = np.linalg.norm(l2)
    cosine = dot_product / (magnitude_X * magnitude_Y)

    similarity_percentage = cosine * 10

    return similarity_percentage


def get_max_similarity(sample_answers, given_answer):
    max_similarity = 0
    for sample in sample_answers:
        essay = sample['essay']
        similarity = cosine_similarity(essay, given_answer)
        max_similarity = max(max_similarity, similarity)
    return max_similarity


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text


def split_keywords(keywords):
    split_words = []
    for keyword in keywords:
        split_words.extend(keyword.split())
    return split_words


def assign_marks_from_keywords(question_id, answer):
    max_marks = 0
    max_keywords = []
    answer = clean_text(answer)

    relevant_keywords = [data for data in keywords_data if data["questionId"] == question_id]

    for question_data in relevant_keywords:
        # Split multi-word keywords into individual words
        all_keywords = split_keywords(question_data["keywords"])
        common_keywords = set(map(str.lower, all_keywords)) & set(answer.split())
        if len(common_keywords) > len(max_keywords):
            max_keywords = common_keywords
            max_marks = question_data["marks"]

    return max_marks, question_data["keywords"]


@api_view(['POST'])
def finish_test(request):
    if request.method == 'POST':
        try:
            language_test_id = request.query_params.get('languageTestId')
            language_test = get_object_or_404(LanguageTest, languageTestId=language_test_id)
            question_answers = QuestionAnswer.objects.filter(languageTestId=language_test_id)
            total_marks = question_answers.aggregate(total_marks=Sum('answerId__marks'))['total_marks']
            language_test.totalMarks = total_marks
            if total_marks >= 60:
                status = 'pass'
            else:
                status = 'fail'
            language_test.status = status
            language_test.save()

            return JsonResponse(
                {'message': 'Test completed successfully.', 'totalMarks': total_marks, 'status': status},
                status=200)

        except LanguageTest.DoesNotExist:
            return JsonResponse({'error': 'Invalid LanguageTest ID'}, status=400)

    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


def predict_score(answer):
    user_input_seq = tokenizer.texts_to_sequences([answer])
    user_input_pad = pad_sequences(user_input_seq, maxlen=maxlen)
    predicted_score = loaded_model.predict(user_input_pad)
    rounded_score = round(predicted_score[0][0])

    return rounded_score


def keyword_matching(answer, keywords_list):
    matching_keywords_in_answer = []
    answer_words_list = answer.split()
    print('keywords_list :', keywords_list)

    try:
        for i in range(len(answer_words_list)):
            new_word_1 = answer_words_list[i] + ' ' + answer_words_list[i + 1] + ' ' + answer_words_list[i + 2] + ' ' + \
                         answer_words_list[i + 3]
            for keyword in keywords_list:
                if keyword == new_word_1:
                    matching_keywords_in_answer.append(keyword)
                    break
            new_word_2 = answer_words_list[i] + ' ' + answer_words_list[i + 1] + ' ' + answer_words_list[i + 2]
            for keyword in keywords_list:
                if keyword == new_word_2:
                    matching_keywords_in_answer.append(keyword)
                    break
            new_word_3 = answer_words_list[i] + ' ' + answer_words_list[i + 1]
            for keyword in keywords_list:
                if keyword == new_word_3:
                    matching_keywords_in_answer.append(keyword)
                    break
            new_word_4 = answer_words_list[i]
            for keyword in keywords_list:
                if keyword == new_word_4:
                    matching_keywords_in_answer.append(keyword)
                    break
    except:
        print("issue")

    return matching_keywords_in_answer


def remove_different_keywords(keyword_list, matching_answer_keywords_list):
    temp_keyword_list = [] + keyword_list
    print('temp_keyword_list : ', temp_keyword_list)
    for word in keyword_list:
        matching_status = 'false'
        for matching_word in matching_answer_keywords_list:
            if word == matching_word:
                matching_status = 'true'
                break
        if matching_status == 'false':
            temp_keyword_list.remove(word)

    print('Updated temp_keyword_list : ', temp_keyword_list)
    return temp_keyword_list


def count_words_in_correct_order(keywords_from_list, keywords_from_answer):
    keywords_from_list_lower = [phrase.lower() for phrase in keywords_from_list]
    keywords_from_answer_lower = [phrase.lower() for phrase in keywords_from_answer]

    correct_order_count = 0

    for i in range(len(keywords_from_list_lower)):
        if i == len(keywords_from_list_lower) - 1:
            left_word_index = keywords_from_answer_lower.index(keywords_from_list_lower[i - 1])
            right_word_index = keywords_from_answer_lower.index(keywords_from_list_lower[i])
            if (right_word_index - left_word_index) == 1:
                correct_order_count = correct_order_count + 1
            break
        left_word_index = keywords_from_answer_lower.index(keywords_from_list_lower[i])
        right_word_index = keywords_from_answer_lower.index(keywords_from_list_lower[i + 1])
        if (right_word_index - left_word_index) == 1:
            correct_order_count = correct_order_count + 1
        elif i != 0:
            left_word_index = keywords_from_answer_lower.index(keywords_from_list_lower[i - 1])
            right_word_index = keywords_from_answer_lower.index(keywords_from_list_lower[i])
            if (right_word_index - left_word_index) == 1:
                correct_order_count = correct_order_count + 1

    return correct_order_count

