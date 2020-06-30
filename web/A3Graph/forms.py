from django import forms

class DocumentForm(forms.Form):
    docfile1 = forms.FileField(
        label='feature',
        help_text='max. 42 megabytes'
    )
    docfile2 = forms.FileField(
        label='structure',
        help_text='max. 42 megabytes'
    )
