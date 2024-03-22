from datetime import timedelta

from django.shortcuts import get_list_or_404
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from common.serializers import StudentSerializer, AddressSerializer, RecruiterSerializer, CV_ProfileSerializer, \
    ObjectiveSerializer, EducationSerializer, SkillSerializer, SocialMediaSerializer, WorkExperienceSerializer, \
    VolunteerExperienceSerializer, ProjectSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Student, Address, Recruiter, CV_Profile, Objective, Education, Skill, SocialMedia, WorkExperience, \
    VolunteerExperience, Project
from django.utils import timezone


# create new user
@api_view(['POST'])
def createUser(request):
    data = request.data.copy()

    print(data)

    # dob birth is mysql date format, so if dob is empty, set it to 2006 january 1st
    if data['dob'] == '':
        data['dob'] = '2006-01-01'

    serializer = StudentSerializer(data=data)

    if serializer.is_valid():
        serializer.save()
        resposeData = {
            'data': serializer.data,
            'message': "User created successfully",
            'status': status.HTTP_201_CREATED
        }
        print(resposeData)
        return Response(resposeData, status=status.HTTP_201_CREATED)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# update user profile by id
@api_view(['PUT'])
def updateUser(request, id):
    try:
        user = Student.objects.get(userID=id)
    except Student.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    data = request.data.copy()
    serializer = StudentSerializer(user, data=data)

    if serializer.is_valid():
        serializer.save()
        responseData = {
            'data': serializer.data,
            'message': 'User updated successfully',
            'status': status.HTTP_200_OK
        }

        return Response(responseData, status=status.HTTP_200_OK)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# delete user by id
def deleteUser(request, id):
    try:
        user = Student.objects.get(userID=id)
    except Student.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    user.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


# check valid user
@api_view(['POST'])
def checkValidUser(request):
    data = request.data.copy()
    email = data['email']
    userPassword = data['userPassword']

    # check if email and password are empty
    if email == '' or userPassword == '':
        responseData = {
            'message': 'Email and password cannot be empty',
            'status': status.HTTP_400_BAD_REQUEST
        }
        return Response(responseData, status=status.HTTP_400_BAD_REQUEST)

    # check if user exists
    user = Student.objects.filter(email=email, userPassword=userPassword)

    # if user exists, return user data
    if user.exists():
        serializer = StudentSerializer(user, many=True)

        responseData = {
            'data': serializer.data[0],
            'message': 'User found',
            'status': status.HTTP_200_OK
        }

        return Response(responseData, status=status.HTTP_200_OK)
    else:
        responseData = {
            'message': 'User not found',
            'status': status.HTTP_404_NOT_FOUND
        }

        return Response(responseData, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET'])
def viewAllUsers(request):
    queryset = get_list_or_404(Student)
    serializer = StudentSerializer(queryset, many=True)
    return Response(serializer.data)


# get user profile details
@api_view(['GET'])
def getUserProfile(request, id):
    try:
        user = Student.objects.get(userID=id)
    except Student.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    serializer = StudentSerializer(user)
    return Response(serializer.data)


@api_view(['GET'])
def getStudentCVProfile(request, id):
    try:
        user = Student.objects.get(userID=id)
        address = Address.objects.get(userID=id)
        cvProfile = CV_Profile.objects.get(userID=id)
        objective = Objective.objects.get(cvID=cvProfile.cvID)
        education = Education.objects.filter(cvID=cvProfile.cvID)
        skill = Skill.objects.filter(cvID=cvProfile.cvID)
        socialMedia = SocialMedia.objects.filter(cvID=cvProfile.cvID)
        workExperience = WorkExperience.objects.filter(cvID=cvProfile.cvID)
        volunteerExperience = VolunteerExperience.objects.filter(cvID=cvProfile.cvID)
        project = Project.objects.filter(cvID=cvProfile.cvID)

        # Serialize the data
        userSerializer = StudentSerializer(user)
        addressSerializer = AddressSerializer(address)
        cvProfileSerializer = CV_ProfileSerializer(cvProfile)
        objectiveSerializer = ObjectiveSerializer(objective)
        educationSerializer = EducationSerializer(education, many=True)
        skillSerializer = SkillSerializer(skill, many=True)
        socialMediaSerializer = SocialMediaSerializer(socialMedia, many=True)
        workExperienceSerializer = WorkExperienceSerializer(workExperience, many=True)
        volunteerExperienceSerializer = VolunteerExperienceSerializer(volunteerExperience, many=True)
        projectSerializer = ProjectSerializer(project, many=True)

        responseData = {
            'data': {
                'student': userSerializer.data,
                'address': addressSerializer.data,
                'cvProfile': cvProfileSerializer.data,
                'objective': objectiveSerializer.data,
                'education': educationSerializer.data,
                'skill': skillSerializer.data,
                'socialMedia': socialMediaSerializer.data,
                'workExperience': workExperienceSerializer.data,
                'volunteerExperience': volunteerExperienceSerializer.data,
                'project': projectSerializer.data
            },
            'status': status.HTTP_200_OK
        }

        return Response(responseData, status=status.HTTP_200_OK)
    except Student.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)


#  update CV_Profile points
@api_view(['PUT'])
def updateStudentPoints(request, id, points):
    try:
        user = Student.objects.get(userID=id)
        cvProfile = CV_Profile.objects.get(userID=id)
    except Student.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    data = request.data.copy()
    serializer = CV_ProfileSerializer(cvProfile, data=data)

    if serializer.is_valid():
        serializer.save()
        responseData = {
            'data': serializer.data,
            'message': 'User points updated successfully',
            'status': status.HTTP_200_OK
        }

        return Response(responseData, status=status.HTTP_200_OK)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# create cv profile
@api_view(['POST'])
def createStudentCVProfile(request):
    data = request.data.copy()
    # update dob in student table
    userID = data['userID']
    dob = data['dob']
    try:
        user = Student.objects.get(userID=userID)
        user.dob = dob
        user.save()

        user_serializer = StudentSerializer(user)
    except Student.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    # insert address details
    addressID = data['addressID']
    userID = data['userID']
    lineOne = data['lineOne']
    lineTwo = data['lineTwo']
    city = data['city']
    postCode = data['postCode']

    # insert address details
    addressSerializer = AddressSerializer(data=data)
    if addressSerializer.is_valid():
        addressSerializer.save()
    else:
        return Response(addressSerializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # insert cv profile details
    cvProfileSerializer = CV_ProfileSerializer(data=data)
    cvID = data['cvID']
    userID = data['userID']
    profile_img = data['profile_img']
    about = data['about']
    points = data['points']

    print(data)

    if cvProfileSerializer.is_valid():
        cvProfileSerializer.save()
        responseData = {
            'data': {
                "student": user_serializer.data,
                "address": addressSerializer.data,
                "cvProfile": cvProfileSerializer.data
            },
            'message': 'CV Profile created successfully',
            'status': status.HTTP_201_CREATED
        }
        return Response(responseData, status=status.HTTP_201_CREATED)
    else:
        return Response(cvProfileSerializer.errors, status=status.HTTP_400_BAD_REQUEST)


def get_cv_ids(request):
    job_title = request.GET.get('job_title')
    duration = request.GET.get('duration')
    skill_name = request.GET.get('skill_name')
    skill_level = request.GET.get('skill_level')
    education_course = request.GET.get('education_course')

    if not job_title or not duration or not skill_name or not skill_level:
        return JsonResponse({'error': 'job_title, duration, skill_name, and skill_level are required parameters'},
                            status=400)

    duration = timedelta(days=int(duration))
    work_experiences = WorkExperience.objects.filter(
        job_title=job_title,
        duration__gte=duration
    )
    cv_ids_from_work_experience = set(work_experiences.values_list('cvID_id', flat=True))
    skills = Skill.objects.filter(
        skill_name=skill_name,
        skill_level=skill_level
    )
    cv_ids_from_skills = set(skills.values_list('cvID_id', flat=True))
    educations = Education.objects.filter(
        education_course=education_course
    )

    # Get the unique CV_IDs from Education
    cv_ids_from_education = set(educations.values_list('cvID_id', flat=True))
    common_cv_ids = cv_ids_from_work_experience.intersection(cv_ids_from_skills, cv_ids_from_education)

    return JsonResponse({'cv_ids': list(common_cv_ids)})
