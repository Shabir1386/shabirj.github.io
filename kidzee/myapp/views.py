from django.shortcuts import render, render_to_response

# Create your views here.
def index(request):
    return render_to_response('index.html')
def submit(request):
    info=request.POST['info']
    return info