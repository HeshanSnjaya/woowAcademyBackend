from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST
import pandas as pd
import joblib
from rest_framework.decorators import api_view

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

