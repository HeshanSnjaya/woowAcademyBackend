from django.urls import path
from . import views

urlpatterns = [
    path('get-random-questions/', views.get_random_questions, name='get_random_questions'),
    path('submit-answer/', views.submit_answer, name='submit_answer'),
    path('finish-test/', views.finish_test, name='finish_test'),
    path('predict_score/', views.predict_score, name='predict_score'),
]