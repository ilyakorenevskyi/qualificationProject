from django.http import Http404, HttpResponseNotAllowed
from django.shortcuts import render, redirect
import neuralNews.generator as gn
import neuralNews.classifier as cl
import tensorflow as tf


def main_page(request):
    return redirect("news_classification")


def news_classification(request):
    if request.method == 'GET':
        return render(request, 'news_classification.html', {})
    elif request.method == 'POST':
        if len(request.POST["text"]) < 100:
            text_class = "Введіть хоча б 100 символів, зараз - " + str(len(request.POST["text"]))
        else:
            classifier = cl.Classifier()
            text_class = classifier.classify(request.POST["text"])
        context = {
            'text': request.POST["text"],
            'result_category': text_class
        }
        return render(request, 'news_classification.html', context)
    return HttpResponseNotAllowed()


def news_generation(request):
    if request.method == 'GET':
        return render(request, 'news_generation.html', {})
    elif request.method == 'POST':
        if len(request.POST["text"]) < 1:
            result_text = "Введіть хоча б 1 початковий символ"
        else:
            generator = gn.Generator(request.POST['category'])
            result_text = generator.generate(request.POST['text'], 200)
        context = {
            'text': request.POST["text"],
            'result_text': result_text
         }
        return render(request, 'news_generation.html', context)
    return HttpResponseNotAllowed()


def feedback(request):
    if request.method == 'POST':
            return render(request, 'news_classification.html', {})
    return HttpResponseNotAllowed()