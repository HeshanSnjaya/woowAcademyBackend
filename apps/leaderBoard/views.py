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

from apps.users.models import CV_Profile, Student, Skill, Education

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
            data = json.loads(request.body.decode('utf-8'))

            given_education = data.get('education')
            given_skill_name = data.get('skill_name')

            top_cv_profiles_with_similarity = get_top_similar_cv_profiles(given_education, given_skill_name)

            serialized_cv_profiles = []
            for cv_profile, similarity in top_cv_profiles_with_similarity:
                serialized_cv_profiles.append({
                    'cvID': cv_profile.cvID,
                    'studentID': cv_profile.studentID.studentID,
                    'username': cv_profile.studentID.userName,
                    'email': cv_profile.studentID.email,
                    'level': cv_profile.studentID.level,
                    'profile_img': cv_profile.profile_img,
                    'about': cv_profile.about,
                    'points': cv_profile.points,
                    'similarity': similarity  # Include similarity value
                })

            return JsonResponse(serialized_cv_profiles, safe=False)
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

    similarity_percentage = cosine * 100

    return similarity_percentage


def get_top_similar_cv_profiles(given_education, given_skill_name):
    similarity_scores = []

    all_cv_profiles = CV_Profile.objects.all()

    for cv_profile in all_cv_profiles:
        education_courses = Education.objects.filter(cvID=cv_profile)

        education_similarity = 0
        for education_course in education_courses:
            education_similarity_temp = cosine_similarity(education_course.education_course.lower(), given_education.lower())
            education_similarity = education_similarity + education_similarity_temp
        if education_courses:
            education_similarity /= len(education_courses)  # Taking average similarity

        skills = Skill.objects.filter(cvID=cv_profile)

        skill_similarity = 0
        for skill in skills:
            skill_similarity += cosine_similarity(skill.skill_name.lower(), given_skill_name.lower())
        if skills:
            skill_similarity /= len(skills)  # Taking average similarity

        total_similarity = (education_similarity + skill_similarity) / 2
        similarity_scores.append((cv_profile, total_similarity))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    top_cv_profiles_with_similarity = similarity_scores[:10]
    return top_cv_profiles_with_similarity