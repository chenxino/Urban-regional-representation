from django import forms

class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Select a file',
        help_text='max. 42 megabytes'
    )

class UserForm(forms.Form):
    # username = forms.CharField()
    # file_field = forms.ImageField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
    img_01 = forms.ImageField(label='第1张图片', )
    # img_02 = forms.ImageField(label='第2张图片', )
