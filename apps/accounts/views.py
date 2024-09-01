from django.contrib.auth import login
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import reverse_lazy
from django.views.generic import CreateView, TemplateView

from .forms import CustomAuthenticationForm, CustomUserCreationForm


class RegisterView(CreateView):
    form_class = CustomUserCreationForm
    template_name = "accounts/register.html"
    success_url = reverse_lazy("home")

    def form_valid(self, form):
        response = super().form_valid(form)
        login(self.request, self.object)
        return response


class CustomLoginView(LoginView):
    form_class = CustomAuthenticationForm
    template_name = "accounts/login.html"

    def get_success_url(self):
        return reverse_lazy("home")


class CustomLogoutView(LogoutView):
    next_page = reverse_lazy("home")


class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = "accounts/profile.html"
