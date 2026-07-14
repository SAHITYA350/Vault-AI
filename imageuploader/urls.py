from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('library/', views.library_view, name='library'),
    path('chat/', views.chat_view, name='chat'),
    path('chat/message/', views.chat_message_api, name='chat_message_api'),
    path('chat/sessions/', views.chat_sessions_api, name='chat_sessions_api'),
    path('chat/session/<int:session_id>/', views.chat_session_detail_api, name='chat_session_detail_api'),
    path('chat/session/create/', views.chat_session_create_api, name='chat_session_create_api'),
    path('chat/session/<int:session_id>/delete/', views.delete_chat_session_api, name='delete_chat_session_api'),
    path('chat/sessions/delete-all/', views.delete_all_chat_sessions_api, name='delete_all_chat_sessions_api'),
    path('graph/', views.graph_view, name='graph'),
    path('graph/data/', views.graph_data_api, name='graph_data_api'),
    path('rechunk/document/', views.rechunk_document_api, name='rechunk_document_api'),
    path('rechunk/all/', views.rechunk_all_api, name='rechunk_all_api'),
    path('studio/', views.studio_view, name='studio'),
    path('studio/generate/', views.studio_generate_api, name='studio_generate_api'),
    path('studio/examiner/', views.studio_examiner_api, name='studio_examiner_api'),
    path('document/<int:doc_id>/delete/', views.delete_document, name='delete_document'),
    path('payment/create-order/', views.payment_create_order, name='payment_create_order'),
    path('payment/verify/', views.payment_verify, name='payment_verify'),
    path('payment/success/', views.payment_success, name='payment_success'),
    path('payment/failure/', views.payment_failure, name='payment_failure'),
    path('payment/change-plan/', views.change_plan_api, name='change_plan_api'),
    path('emails/', views.email_logs_view, name='email_logs_view'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)