import json

from django.db import transaction
from django.db.models import Sum
from django.http import JsonResponse
from .models import LanguageTest, Question, Answer, LanguageTestQuestion, QuestionAnswer, StudentLanguageTest
from django.shortcuts import get_object_or_404

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from rest_framework.decorators import api_view
import random

from ..users.models import Student

loaded_model = load_model('D:\DjangoProjects\woow-academy-backend\my_model.h5')
data = pd.read_csv("D:\DjangoProjects\woow-academy-backend\ielts_writing_dataset.csv")
X = data['Essay']
y = data['Overall']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
maxlen = len(max(X_seq, key=len))


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

            rounded_score = predict_score(answer_description)
            additional_points = 0
            if 6 <= duration <= 9:
                if duration <= 6:
                    additional_points = 0.5
                elif 6 < duration <= 8:
                    additional_points = 0.25

            final_marks = rounded_score + additional_points
            print('rounded score : ' + str(rounded_score))
            print('additional point : ' + str(additional_points))
            print('final marks : ' + str(final_marks))

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
                language_test.status = 'pass'
            else:
                language_test.status = 'fail'
            language_test.save()

            return JsonResponse({'message': 'Test completed successfully.', 'totalMarks': total_marks}, status=200)

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
