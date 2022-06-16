from django import forms


class NewsClassificationForm(forms.ModelForm):
       articletext = forms.body = forms.CharField(widget=forms.Textarea(attrs={'placeholder':'введи новину'}))