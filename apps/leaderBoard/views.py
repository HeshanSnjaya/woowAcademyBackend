import json

from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
import pandas as pd
import joblib
from rest_framework.decorators import api_view

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv('D:\DjangoProjects\woow-academy-backend\job_dataset_3.csv')

from apps.users.models import CV_Profile, Student

best_clf = joblib.load('D:\DjangoProjects\woow-academy-backend\decision_tree_model_2.pkl')
imputer = joblib.load('D:\DjangoProjects\woow-academy-backend\decision_imputer_2.pkl')


@require_GET
def predict_level(request):
    job_title = request.GET.get('job_title', '')
    education = request.GET.get('education', '')
    skills_name = request.GET.get('skills_name', '')
    min_experience = int(request.GET.get('min_experience', 0))

    random_data = pd.DataFrame({
        'job_title': [job_title],
        'education': [education],
        'skills_name': [skills_name],
        'min_experience': min_experience
    })

    random_data_encoded = pd.get_dummies(random_data)
    random_data_imputed = random_data_encoded.reindex(columns=imputer.get_feature_names_out(), fill_value=0)
    random_data_imputed = imputer.transform(random_data_imputed)
    predictions = best_clf.predict(random_data_imputed)

    return JsonResponse({'predicted_level': predictions[0]})


# Get request to get the list of leader board in reverse order
@require_GET
def get_cv_profiles(request):
    cv_profiles = CV_Profile.objects.select_related('studentID').order_by('-points')

    serialized_cv_profiles = [{
        'cvID': profile.cvID,
        'studentID': profile.studentID.studentID,
        'username': profile.studentID.userName,
        'email': profile.studentID.email,
        'level': profile.studentID.level,
        'profile_img': profile.profile_img,
        'about': profile.about,
        'points': profile.points
    } for profile in cv_profiles]

    return JsonResponse(serialized_cv_profiles, safe=False)


# View to person in leaderboard by username
@require_GET
def get_cv_profile_by_username(request, username):
    try:
        student = Student.objects.get(userName=username)

        cv_profiles = CV_Profile.objects.filter(studentID=student).order_by('-points')

        serialized_cv_profiles = [{
            'cvID': profile.cvID,
            'studentID': profile.studentID.studentID,
            'username': profile.studentID.userName,
            'email': profile.studentID.email,
            'level': profile.studentID.level,
            'profile_img': profile.profile_img,
            'about': profile.about,
            'points': profile.points
        } for profile in cv_profiles]

        return JsonResponse(serialized_cv_profiles, safe=False)

    except Student.DoesNotExist:
        return JsonResponse([], safe=False)

@require_GET
def get_cv_profiles_by_level(request, level):
    try:
        students = Student.objects.filter(level=level)
        cv_profiles = CV_Profile.objects.filter(studentID__in=students).order_by('-points')
        serialized_cv_profiles = [{
            'cvID': profile.cvID,
            'studentID': profile.studentID.studentID,
            'username': profile.studentID.userName,
            'email': profile.studentID.email,
            'level': profile.studentID.level,
            'profile_img': profile.profile_img,
            'about': profile.about,
            'points': profile.points
        } for profile in cv_profiles]

        return JsonResponse(serialized_cv_profiles, safe=False)

    except Student.DoesNotExist:
        return JsonResponse([], safe=False)


# top 10 usernames with similarity
@api_view(['GET'])
def get_top_names(request):
    if request.method == 'GET':
        try:
            if request.body:
                data = json.loads(request.body.decode('utf-8'))

                given_education = data.get('education')
                given_skill_name = data.get('skill_name')

                top_names = get_top_similar_user_names(given_education, given_skill_name)

                return JsonResponse({'top_names': top_names})
            else:
                return JsonResponse({'error': 'Request body is empty'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

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


def get_top_similar_user_names(given_education, given_skill_name):
    similarity_scores = []

    for index, row in df.iterrows():
        education_similarity = cosine_similarity(row['education'], given_education)
        skill_similarity = cosine_similarity(row['skills_name'], given_skill_name)
        total_similarity = (education_similarity + skill_similarity) / 2
        similarity_scores.append((row['userName'], total_similarity))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_names_with_similarity = [(row[0], row[1]) for row in similarity_scores[:10]]
    return top_names_with_similarity
