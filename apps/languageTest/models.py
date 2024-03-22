from django.db import models
from apps.users.models import Student


class LanguageTest(models.Model):
    languageTestId = models.AutoField(primary_key=True)
    totalMarks = models.FloatField(default=0)
    status = models.CharField(max_length=100, default='pending')
    questions = models.ManyToManyField('Question', through='LanguageTestQuestion')

    class Meta:
        db_table = 'LanguageTest'


class StudentLanguageTest(models.Model):
    studentID = models.ForeignKey(Student, on_delete=models.CASCADE)
    languageTestId = models.ForeignKey(LanguageTest, on_delete=models.CASCADE)

    class Meta:
        db_table = 'StudentLanguageTest'


class Question(models.Model):
    questionId = models.AutoField(primary_key=True)
    questionDesc = models.TextField()

    class Meta:
        db_table = 'Question'


class Answer(models.Model):
    answerId = models.AutoField(primary_key=True)
    answerDesc = models.TextField()
    marks = models.FloatField(default=0)
    additionalPoints = models.FloatField(default=0)

    class Meta:
        db_table = 'Answer'


class LanguageTestQuestion(models.Model):
    languageTestId = models.ForeignKey(LanguageTest, on_delete=models.CASCADE)
    questionId = models.ForeignKey(Question, on_delete=models.CASCADE)

    class Meta:
        db_table = 'LanguageTestQuestion'


class QuestionAnswer(models.Model):
    questionId = models.ForeignKey(Question, on_delete=models.CASCADE)
    languageTestId = models.ForeignKey(LanguageTest, on_delete=models.CASCADE)
    answerId = models.ForeignKey(Answer, on_delete=models.CASCADE)
    duration = models.FloatField(default=0)

    class Meta:
        db_table = 'QuestionAnswer'
