from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_pic = models.ImageField(upload_to="profile_pics", blank=True, null=True)
    storage_limit = models.BigIntegerField(default=104857600) # 100 MB default
    storage_used = models.BigIntegerField(default=0)
    is_premium = models.BooleanField(default=False)
    auto_pay = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username}'s Profile"

    @property
    def storage_percentage(self):
        if self.storage_limit > 0:
            return min(100, int((self.storage_used / self.storage_limit) * 100))
        return 0

    @property
    def storage_used_formatted(self):
        # Format size in KB or MB
        size = self.storage_used
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @property
    def storage_limit_formatted(self):
        size = self.storage_limit
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


class Document(models.Model):
    FILE_TYPES = [
        ('image', 'Image'),
        ('pdf', 'PDF'),
        ('docx', 'Word Document'),
        ('text', 'Text/Markdown'),
        ('zip', 'Archive'),
        ('document', 'Other Document'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    file = models.FileField(upload_to="vault/")
    name = models.CharField(max_length=255, blank=True)
    file_type = models.CharField(max_length=20, choices=FILE_TYPES, default='document')
    file_size = models.BigIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Placeholder fields for AI processing phases
    extracted_text = models.TextField(blank=True, default='')
    ai_caption = models.TextField(blank=True, default='')
    ai_summary = models.TextField(blank=True, default='')
    ai_tags = models.TextField(blank=True, default='') # Comma separated tags
    embedding = models.JSONField(blank=True, null=True)
    ai_quiz = models.JSONField(blank=True, null=True)
    ai_flashcards = models.JSONField(blank=True, null=True)
    # Whether deep semantic chunking has been run
    is_chunked = models.BooleanField(default=False)

    @property
    def file_size_formatted(self):
        # Format size in KB or MB
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def __str__(self):
        return self.name or self.file.name

    def save(self, *args, **kwargs):
        # Set file name if not provided
        if not self.name and self.file:
            self.name = self.file.name.split('/')[-1]

        # Calculate file size
        if self.file and not self.file_size:
            self.file_size = self.file.size

        # Automatically determine file category based on extension
        if self.file:
            ext = self.file.name.split('.')[-1].lower()
            if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg']:
                self.file_type = 'image'
            elif ext == 'pdf':
                self.file_type = 'pdf'
            elif ext in ['doc', 'docx']:
                self.file_type = 'docx'
            elif ext in ['txt', 'md', 'csv', 'py', 'js', 'json', 'html', 'css']:
                self.file_type = 'text'
            elif ext in ['zip', 'rar', 'tar', 'gz', '7z']:
                self.file_type = 'zip'
            else:
                self.file_type = 'document'

        # Calculate difference in storage usage if file is new or updated
        old_size = 0
        if self.pk:
            try:
                old_doc = Document.objects.get(pk=self.pk)
                old_size = old_doc.file_size
            except Document.DoesNotExist:
                pass

        super().save(*args, **kwargs)

        # Update UserProfile storage used
        profile = self.user.userprofile
        profile.storage_used = profile.storage_used - old_size + self.file_size
        profile.save()


class DocumentChunk(models.Model):
    """Stores a single semantic chunk of a document — page-level or paragraph-level."""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    page_number = models.IntegerField(default=1)        # 1-indexed page
    chunk_index = models.IntegerField(default=0)        # Position within the page
    text = models.TextField()                           # The actual chunk text (300-600 chars)
    embedding = models.JSONField(blank=True, null=True) # 384d semantic embedding
    concept_label = models.CharField(max_length=200, blank=True, default='') # AI concept name
    importance_score = models.FloatField(default=0.5)  # 0-1 relevance weight
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['page_number', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'page_number']),
        ]

    def __str__(self):
        return f"{self.document.name} | P{self.page_number}C{self.chunk_index} | {self.concept_label or 'chunk'}"


class Collection(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='collections')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, default='')
    documents = models.ManyToManyField(Document, related_name='collections', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    title = models.CharField(max_length=255, default='New Conversation')
    document = models.ForeignKey(Document, on_delete=models.SET_NULL, null=True, blank=True, related_name='chat_sessions')
    tags = models.CharField(max_length=255, blank=True, default='') # Space or comma separated tags for search
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    sender = models.CharField(max_length=10, choices=[('user', 'User'), ('ai', 'AI')])
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.content[:30]}"


# Signal receivers to maintain storage usage and auto-create profiles

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    # Ensure profile exists before saving
    profile, created = UserProfile.objects.get_or_create(user=instance)
    profile.save()

@receiver(post_delete, sender=Document)
def delete_document_file_and_update_storage(sender, instance, **kwargs):
    # Delete the physical file from disk
    if instance.file:
        instance.file.delete(save=False)
        
    # Subtract storage size from user profile
    try:
        profile = instance.user.userprofile
        profile.storage_used = max(0, profile.storage_used - instance.file_size)
        profile.save()
    except (User.DoesNotExist, UserProfile.DoesNotExist):
        pass