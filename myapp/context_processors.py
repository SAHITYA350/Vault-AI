def profile_context(request):
    if request.user.is_authenticated:
        try:
            from .models import UserProfile
            from allauth.socialaccount.models import SocialAccount
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            # Fetch Google profile picture URL
            google_avatar = None
            try:
                social = SocialAccount.objects.filter(user=request.user, provider='google').first()
                if social:
                    extra = social.extra_data
                    google_avatar = extra.get('picture') or extra.get('avatar_url')
            except Exception:
                pass
            return {'profile': profile, 'google_avatar': google_avatar}
        except Exception as e:
            print(f"Error in profile context processor: {e}")
    return {}
