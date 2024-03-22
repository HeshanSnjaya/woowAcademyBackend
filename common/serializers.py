from rest_framework import serializers
from apps.users.models import CV_Profile, Objective, Education, Skill, SocialMedia, WorkExperience, VolunteerExperience, \
    Project, Student, Address, Recruiter
from apps.courses.models import Course
from apps.languageTest.models import LanguageTest, Question, Answer


class StudentSerializer(serializers.ModelSerializer):
    userID = serializers.CharField(max_length=50)
    firstName = serializers.CharField(max_length=255)
    lastName = serializers.CharField(max_length=255)
    userName = serializers.CharField(max_length=255)
    email = serializers.CharField(max_length=255)
    phone = serializers.CharField(max_length=255)
    dob = serializers.DateField(default='2006-01-01')
    userPassword = serializers.CharField(max_length=255)

    class Meta:
        model = Student
        fields = '__all__'


class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Course
        fields = ['id', 'courseName', 'description']


class LanguageTestSerializer(serializers.ModelSerializer):
    class Meta:
        model = LanguageTest
        fields = '__all__'


class QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Question
        fields = '__all__'


class AnswerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Answer
        fields = '__all__'


class AddressSerializer(serializers.ModelSerializer):
    class Meta:
        model = Address
        fields = '__all__'


class RecruiterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recruiter
        fields = '__all__'


class CV_ProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = CV_Profile
        fields = '__all__'


class ObjectiveSerializer(serializers.ModelSerializer):
    class Meta:
        model = Objective
        fields = '__all__'


class EducationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Education
        fields = '__all__'


class SkillSerializer(serializers.ModelSerializer):
    class Meta:
        model = Skill
        fields = '__all__'


class SocialMediaSerializer(serializers.ModelSerializer):
    class Meta:
        model = SocialMedia
        fields = '__all__'


class WorkExperienceSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkExperience
        fields = '__all__'


class VolunteerExperienceSerializer(serializers.ModelSerializer):
    class Meta:
        model = VolunteerExperience
        fields = '__all__'


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = '__all__'
