import base64
import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st
from PIL import Image

from app.logging_config import configure_logging, emit_audit_event
from app.pipelines.fraud_detector import detect_fraud_signals
from app.pipelines.listing_quality import analyze_listing_quality
from app.pipelines.price_recommender import recommend_price
from app.pipelines.trust_score import compute_trust_score
from app.security import (
    MAX_DESCRIPTION_LENGTH,
    MAX_LOCATION_LENGTH,
    MAX_PRICE,
    MAX_TITLE_LENGTH,
    RateLimiter,
    get_auth_credentials,
    get_client_identity,
    validate_listing_input,
)


def get_secret_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        secret_value = str(st.secrets.get(name, "")).strip()
        return secret_value
    except Exception:
        return ""


APP_VERSION = "2026.07.07"


def hash_password(password: str) -> str:
    salt = get_secret_value("APP_AUTH_SALT") or "local-dev-salt"
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def verify_password(password: str, expected_hash: str) -> bool:
    return bool(password) and hash_password(password) == expected_hash


def is_contact_verified(account: dict[str, Any]) -> bool:
    return bool(account.get("verified_email") or account.get("verified_phone"))


def redact_sensitive_text(value: str) -> str:
    if not value:
        return value

    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", value)
    text = re.sub(r"\+?\d[\d\s().-]{7,}\d", "[redacted-phone]", text)
    return text


def build_verification_code() -> str:
    return f"{secrets.randbelow(900000) + 100000}"


def ensure_default_account(expected_username: str, expected_password: str) -> None:
    accounts = st.session_state.accounts
    if expected_username not in accounts:
        accounts[expected_username] = {
            "password_hash": hash_password(expected_password),
            "email": "",
            "phone": "",
            "verified_email": True,
            "verified_phone": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def account_exists(username: str) -> bool:
    return username in st.session_state.accounts


def authenticate_account(username: str, password: str) -> bool:
    account = st.session_state.accounts.get(username)
    if not account:
        return False

    if not verify_password(password, str(account.get("password_hash", ""))):
        return False

    return is_contact_verified(account)


def create_account(username: str, password: str, email: str, phone: str) -> None:
    st.session_state.accounts[username] = {
        "password_hash": hash_password(password),
        "email": email,
        "phone": phone,
        "verified_email": False,
        "verified_phone": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_remembered_credentials(client_id: str, username: str, password: str) -> None:
    st.session_state.remembered_credentials_by_client[client_id] = {
        "username": username,
        "password": password,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def clear_remembered_credentials(client_id: str) -> None:
    if client_id in st.session_state.remembered_credentials_by_client:
        del st.session_state.remembered_credentials_by_client[client_id]


def get_remembered_credentials(client_id: str) -> dict[str, str]:
    remembered = st.session_state.remembered_credentials_by_client.get(client_id, {})
    return {
        "username": str(remembered.get("username", "")),
        "password": str(remembered.get("password", "")),
    }


def check_for_available_update() -> dict[str, Any]:
    repo = get_secret_value("APP_GITHUB_REPO") or "danielmacharia172-dot/Online-market-listing-intelligence"
    deployed_commit = get_secret_value("APP_BUILD_COMMIT") or "unknown"

    result: dict[str, Any] = {
        "status": "unknown",
        "message": "Update check not run yet.",
        "latest_commit": "",
        "deployed_commit": deployed_commit,
        "update_needed": False,
    }

    try:
        response = requests.get(f"https://api.github.com/repos/{repo}/commits/main", timeout=10)
        if response.status_code >= 400:
            result["status"] = "error"
            result["message"] = "Could not check GitHub for updates."
            return result

        payload = response.json()
        latest_commit = str(payload.get("sha", ""))[:12]
        result["latest_commit"] = latest_commit

        if deployed_commit != "unknown" and latest_commit and latest_commit != deployed_commit[:12]:
            result["status"] = "update-available"
            result["update_needed"] = True
            result["message"] = "A newer GitHub commit is available. Approval is required before updating."
        else:
            result["status"] = "up-to-date"
            result["message"] = "No required update detected."

        return result
    except Exception:
        result["status"] = "error"
        result["message"] = "Update check failed due to a network/runtime issue."
        return result


def init_state() -> None:
    defaults = {
        "authenticated": False,
        "current_user": "",
        "pending_login_user": "",
        "needs_upload_resume_choice": False,
        "accounts": {},
        "pending_verifications": {},
        "remembered_credentials_by_client": {},
        "update_check_result": {
            "status": "unknown",
            "message": "Update check not run yet.",
            "latest_commit": "",
            "deployed_commit": "unknown",
            "update_needed": False,
        },
        "listing_field_errors": {},
        "buyer_reviews_by_listing": {},
        "analyzed_listings": {},
        "last_listing_key": None,
        "account_profiles": {},
        "password_overrides": {},
        "messages": [],
        "detected_location": "",
        "listing_drafts": {},
        "active_listing_draft": {
            "title": "",
            "description": "",
            "price": 699,
            "location": "",
            "photo_views": {},
            "uploaded_photos": [],
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_default_listing_draft(location_hint: str = "") -> dict[str, Any]:
    return {
        "title": "",
        "description": "",
        "price": 699,
        "location": location_hint,
        "photo_views": {},
        "uploaded_photos": [],
    }


def load_user_listing_draft(username: str, location_hint: str = "") -> None:
    stored = st.session_state.listing_drafts.get(username)
    if stored:
        st.session_state.active_listing_draft = {
            "title": str(stored.get("title", "")),
            "description": str(stored.get("description", "")),
            "price": int(stored.get("price", 699)),
            "location": str(stored.get("location", location_hint)),
            "photo_views": dict(stored.get("photo_views", {})),
            "uploaded_photos": list(stored.get("uploaded_photos", [])),
        }
    else:
        st.session_state.active_listing_draft = get_default_listing_draft(location_hint)


def save_listing_draft_for_user(username: str, draft: dict[str, Any]) -> None:
    st.session_state.listing_drafts[username] = {
        "title": str(draft.get("title", "")),
        "description": str(draft.get("description", "")),
        "price": int(draft.get("price", 699)),
        "location": str(draft.get("location", "")),
        "photo_views": dict(draft.get("photo_views", {})),
        "uploaded_photos": list(draft.get("uploaded_photos", [])),
    }


def clear_listing_draft_for_user(username: str, location_hint: str = "") -> None:
    st.session_state.listing_drafts[username] = get_default_listing_draft(location_hint)
    st.session_state.active_listing_draft = get_default_listing_draft(location_hint)


def analyze_uploaded_photo(uploaded_photo: Any, view_label: str) -> dict[str, Any]:
    uploaded_photo.seek(0)
    image = Image.open(uploaded_photo)
    width, height = image.size
    file_size_kb = round(len(uploaded_photo.getvalue()) / 1024, 2)
    megapixels = round((width * height) / 1_000_000, 2)
    uploaded_photo.seek(0)
    return {
        "width": width,
        "height": height,
        "megapixels": megapixels,
        "format": image.format or "unknown",
        "mode": image.mode,
        "file_size_kb": file_size_kb,
        "view": view_label,
    }


def build_listing_key(title: str, location: str, owner: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_title = re.sub(r"\s+", "-", title.strip().lower())
    safe_location = re.sub(r"\s+", "-", location.strip().lower())
    safe_owner = re.sub(r"\s+", "-", owner.strip().lower())
    return f"{safe_title}|{safe_location}|{safe_owner}|{stamp}"


def listing_label(title: str, location: str) -> str:
    return f"{title} ({location})"


def get_effective_password(username: str, default_password: str) -> str:
    override = st.session_state.password_overrides.get(username)
    return override if override else default_password


def infer_condition(description: str, photo_insights: list[dict[str, Any]]) -> str:
    text = description.lower()
    if any(term in text for term in ["new", "sealed", "unused", "mint"]):
        return "Excellent"
    if any(term in text for term in ["good", "works", "clean", "lightly used"]):
        return "Good"
    if any(term in text for term in ["fair", "scratch", "wear", "older"]):
        return "Fair"

    avg_mp = 0.0
    if photo_insights:
        avg_mp = sum(item["megapixels"] for item in photo_insights) / len(photo_insights)

    if avg_mp >= 4:
        return "Good"
    if avg_mp >= 1.5:
        return "Fair"
    return "Unknown"


def extract_year(text: str) -> str:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return match.group(1) if match else "Unknown"


def estimate_market_range(title: str, condition: str) -> tuple[int, int]:
    text = title.lower()
    baseline = 200

    category_prices = {
        "iphone": 550,
        "samsung": 420,
        "macbook": 900,
        "laptop": 650,
        "playstation": 360,
        "xbox": 320,
        "sofa": 500,
        "couch": 520,
        "bike": 300,
        "camera": 450,
        "watch": 250,
    }

    for keyword, value in category_prices.items():
        if keyword in text:
            baseline = value
            break

    multipliers = {
        "Excellent": 1.0,
        "Good": 0.85,
        "Fair": 0.65,
        "Unknown": 0.75,
    }
    adjusted = baseline * multipliers.get(condition, 0.75)
    low = int(round(adjusted * 0.9))
    high = int(round(adjusted * 1.15))
    return low, high


def generate_ai_suggestion(
    title: str,
    description: str,
    location: str,
    current_price: float,
    photo_insights: list[dict[str, Any]],
    pipeline_price: int,
) -> dict[str, Any]:
    year = extract_year(f"{title} {description}")
    condition = infer_condition(description, photo_insights)
    market_low, market_high = estimate_market_range(title, condition)

    blended_suggested = int(round((pipeline_price + ((market_low + market_high) / 2)) / 2))
    product_name = title.strip().title() if title.strip() else "Unknown product"

    description_lines = [
        f"Product name: {product_name}",
        f"Estimated production year: {year}",
        f"Condition assessment: {condition}",
        f"Location context: {location}",
    ]

    if photo_insights:
        view_summary = ", ".join(item["view"] for item in photo_insights)
        description_lines.append(f"Photo coverage: {view_summary}")

    description_lines.append(
        f"Suggested market price range: ${market_low} - ${market_high}; recommended list price: ${blended_suggested}."
    )

    return {
        "product_name": product_name,
        "year": year,
        "condition": condition,
        "market_low": market_low,
        "market_high": market_high,
        "suggested_price": blended_suggested,
        "generated_description": "\n".join(description_lines),
        "confidence": "medium",
        "source": "heuristic",
        "model": "local-rules",
        "issues_to_edit": [],
    }


def image_to_data_url(uploaded_photo: Any) -> str:
    raw_bytes = uploaded_photo.getvalue()
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }
    file_name = str(getattr(uploaded_photo, "name", "")).lower()
    ext = file_name.split(".")[-1] if "." in file_name else "jpg"
    mime_type = mime_map.get(ext, "image/jpeg")
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def sanitize_ai_vision_output(parsed: dict[str, Any], heuristic_fallback: dict[str, Any]) -> dict[str, Any]:
    def as_int(value: Any, default: int) -> int:
        try:
            return int(round(float(value)))
        except Exception:
            return default

    product_name = str(parsed.get("product_name", heuristic_fallback["product_name"]))
    year = str(parsed.get("year", heuristic_fallback["year"]))
    condition = str(parsed.get("condition", heuristic_fallback["condition"]))
    market_low = as_int(parsed.get("market_low"), heuristic_fallback["market_low"])
    market_high = as_int(parsed.get("market_high"), heuristic_fallback["market_high"])
    suggested_price = as_int(parsed.get("suggested_price"), heuristic_fallback["suggested_price"])
    generated_description = str(parsed.get("generated_description", heuristic_fallback["generated_description"]))
    confidence = str(parsed.get("confidence", "medium"))

    issues_raw = parsed.get("issues_to_edit", [])
    if isinstance(issues_raw, list):
        issues_to_edit = [str(item) for item in issues_raw][:10]
    else:
        issues_to_edit = []

    if market_low > market_high:
        market_low, market_high = market_high, market_low

    return {
        "product_name": product_name,
        "year": year,
        "condition": condition,
        "market_low": max(1, market_low),
        "market_high": max(1, market_high),
        "suggested_price": max(1, suggested_price),
        "generated_description": generated_description,
        "confidence": confidence,
        "source": "vision-model",
        "model": parsed.get("model", "openai-compatible-vision"),
        "issues_to_edit": issues_to_edit,
    }


def generate_ai_suggestion_with_optional_vision(
    title: str,
    description: str,
    location: str,
    current_price: float,
    photo_insights: list[dict[str, Any]],
    pipeline_price: int,
    uploaded_photos: list[Any],
    photo_views: list[str],
) -> dict[str, Any]:
    heuristic = generate_ai_suggestion(
        title=title,
        description=description,
        location=location,
        current_price=current_price,
        photo_insights=photo_insights,
        pipeline_price=pipeline_price,
    )

    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key or not uploaded_photos:
        return heuristic

    model = get_secret_value("OPENAI_MODEL") or "gpt-4o-mini"
    base_url = get_secret_value("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    try:
        image_blocks = []
        max_images = min(len(uploaded_photos), 6)
        for idx in range(max_images):
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(uploaded_photos[idx])},
                }
            )

        views_text = ", ".join(photo_views[:max_images]) if photo_views else "unspecified"
        safe_title = redact_sensitive_text(title)
        safe_description = redact_sensitive_text(description)
        safe_location = redact_sensitive_text(location)
        prompt = (
            "You are a product listing analyst. Analyze the provided product photos and listing context. "
            "Return JSON only with keys: product_name, year, condition, market_low, market_high, "
            "suggested_price, generated_description, confidence, issues_to_edit, model. "
            "Condition must be one of: Excellent, Good, Fair, Poor, Unknown. "
            "generated_description must be practical and specific for resale marketplaces. "
            "issues_to_edit should be an array of concise fixes for the lister. "
            f"Context: title={safe_title}; description={safe_description}; location={safe_location}; current_price={current_price}; "
            f"photo_views={views_text}."
        )

        payload = {
            "model": model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + image_blocks,
                }
            ],
        }

        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=35,
        )
        if response.status_code >= 400:
            return heuristic

        body = response.json()
        text_content = str(body["choices"][0]["message"]["content"])
        parsed = parse_json_object(text_content)
        if not parsed:
            return heuristic

        parsed.setdefault("model", model)
        return sanitize_ai_vision_output(parsed, heuristic)
    except Exception:
        return heuristic


def suggest_location_from_ip(client_id: str) -> str:
    if not client_id or client_id == "local":
        return ""

    try:
        response = requests.get(f"https://ipapi.co/{client_id}/json/", timeout=4)
        if response.status_code != 200:
            return ""
        payload = response.json()
        city = str(payload.get("city", "")).strip()
        region = str(payload.get("region", "")).strip()
        country = str(payload.get("country_name", "")).strip()
        parts = [part for part in [city, region, country] if part]
        return ", ".join(parts)
    except Exception:
        return ""


def render_reviews_section(logger: Any, client_id: str) -> None:
    reviews_by_listing = st.session_state.buyer_reviews_by_listing
    listings = st.session_state.analyzed_listings

    st.markdown("---")
    st.header("Buyer Review Platform")
    st.caption("Collect buyer feedback for each listing so future shoppers can evaluate seller trust and listing quality.")

    listing_options = []
    for key, item in listings.items():
        listing_options.append({"key": key, "label": listing_label(item["title"], item["location"])})

    default_index = 0
    if st.session_state.last_listing_key is not None:
        for idx, option in enumerate(listing_options):
            if option["key"] == st.session_state.last_listing_key:
                default_index = idx
                break

    selected_listing_key = None
    selected_listing_title = ""
    selected_listing_location = ""

    if listing_options:
        selected_label = st.selectbox(
            "Listing to review",
            options=[item["label"] for item in listing_options],
            index=default_index,
        )
        for item in listing_options:
            if item["label"] == selected_label:
                selected_listing_key = item["key"]
                selected_listing_title = str(listings[item["key"]]["title"])
                selected_listing_location = str(listings[item["key"]]["location"])
                break
    else:
        st.info("Analyze a listing first, then buyers can post reviews for that listing.")
        return

    sort_order = st.selectbox("Sort reviews", ["Most recent", "Highest rating", "Lowest rating"])
    minimum_rating = st.slider("Minimum rating filter", min_value=1, max_value=5, value=1)

    with st.form("review_form"):
        reviewer = st.text_input("Buyer name", placeholder="Alex")
        rating = st.slider("Rating", min_value=1, max_value=5, value=5)
        review_text = st.text_area("Review", placeholder="Quick pickup, item exactly as described.")
        review_submitted = st.form_submit_button("Post review")

    if review_submitted:
        clean_reviewer = reviewer.strip()
        clean_review = review_text.strip()

        if not clean_reviewer or not clean_review:
            st.error("Buyer name and review text are required.")
        else:
            review_record = {
                "buyer": clean_reviewer,
                "rating": rating,
                "review": clean_review,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "listing_title": selected_listing_title,
                "listing_location": selected_listing_location,
            }
            bucket = reviews_by_listing.setdefault(selected_listing_key, [])
            bucket.insert(0, review_record)
            emit_audit_event(
                logger,
                "review_posted",
                {
                    "client": client_id,
                    "rating": rating,
                    "listing_key": selected_listing_key,
                },
            )
            st.success("Review posted.")

    reviews = reviews_by_listing.get(selected_listing_key, [])
    reviews = [item for item in reviews if item["rating"] >= minimum_rating]

    if sort_order == "Highest rating":
        reviews = sorted(reviews, key=lambda item: item["rating"], reverse=True)
    elif sort_order == "Lowest rating":
        reviews = sorted(reviews, key=lambda item: item["rating"])

    if not reviews:
        st.info("No reviews match this listing/filter yet. Post a review to get started.")
        return

    average_rating = round(sum(item["rating"] for item in reviews) / len(reviews), 2)
    col1, col2 = st.columns(2)
    col1.metric("Average buyer rating", f"{average_rating}/5")
    col2.metric("Reviews shown", str(len(reviews)))

    st.write(f"Reviews for: **{listing_label(selected_listing_title, selected_listing_location)}**")

    st.markdown("### Recent reviews")
    for item in reviews[:20]:
        stars = "*" * item["rating"]
        st.markdown(f"**{item['buyer']}** ({item['rating']}/5) {stars}")
        st.write(item["review"])


def build_listing_field_errors(title: str, description: str, price: Any, location: str) -> dict[str, str]:
    errors: dict[str, str] = {}

    if not isinstance(title, str) or not title.strip():
        errors["title"] = "Listing title is required."
    elif len(title.strip()) > MAX_TITLE_LENGTH:
        errors["title"] = f"Listing title must be at most {MAX_TITLE_LENGTH} characters."

    if not isinstance(description, str) or not description.strip():
        errors["description"] = "Description is required."
    elif len(description.strip()) > MAX_DESCRIPTION_LENGTH:
        errors["description"] = f"Description must be at most {MAX_DESCRIPTION_LENGTH} characters."

    if not isinstance(location, str) or not location.strip():
        errors["location"] = "Location is required."
    elif len(location.strip()) > MAX_LOCATION_LENGTH:
        errors["location"] = f"Location must be at most {MAX_LOCATION_LENGTH} characters."

    try:
        numeric_price = float(price)
        if numeric_price < 0:
            errors["price"] = "Price cannot be negative."
        elif numeric_price > MAX_PRICE:
            errors["price"] = f"Price must be less than or equal to {MAX_PRICE}."
    except (TypeError, ValueError):
        errors["price"] = "Price must be numeric."

    combined_text = f"{title}{description}{location}"
    if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in combined_text):
        control_error = "Input contains unsupported control characters."
        for field_name in ("title", "description", "location"):
            errors.setdefault(field_name, control_error)

    return errors


def render_listing_field_highlights(errors: dict[str, str]) -> None:
    if not errors:
        return

    selectors = {
        "title": "input[aria-label='Listing title']",
        "description": "textarea[aria-label='Description']",
        "price": "input[aria-label='Price']",
        "location": "input[aria-label='Location']",
    }

    css_rules = []
    for field, selector in selectors.items():
        if field in errors:
            css_rules.append(
                f"{selector} {{ border: 2px solid #ff4b4b !important; box-shadow: 0 0 0 1px rgba(255, 75, 75, 0.35) !important; }}"
            )

    if css_rules:
        st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)


def render_account_security_panel(logger: Any) -> None:
    st.sidebar.header("Account and Security")
    st.sidebar.write(f"Signed in as: **{st.session_state.current_user}**")
    st.sidebar.caption("Privacy mode keeps sensitive personal details out of external AI prompts.")

    profiles = st.session_state.account_profiles
    profile = profiles.get(st.session_state.current_user, {"email": "", "phone": ""})

    with st.sidebar.form("link_contact_form"):
        email = st.text_input("Linked email", value=profile.get("email", ""), placeholder="name@example.com")
        phone = st.text_input("Linked phone", value=profile.get("phone", ""), placeholder="+1 555 123 4567")
        save_contact = st.form_submit_button("Save recovery contacts")

    if save_contact:
        clean_email = email.strip()
        clean_phone = phone.strip()
        profiles[st.session_state.current_user] = {"email": clean_email, "phone": clean_phone}
        if st.session_state.current_user in st.session_state.accounts:
            st.session_state.accounts[st.session_state.current_user]["email"] = clean_email
            st.session_state.accounts[st.session_state.current_user]["phone"] = clean_phone
        emit_audit_event(logger, "backup_contact_updated", {"user": st.session_state.current_user})
        st.sidebar.success("Recovery contacts saved.")

    with st.sidebar.form("change_password_form"):
        current_password = st.text_input("Current password", type="password")
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        do_change = st.form_submit_button("Change password")

    if do_change:
        account = st.session_state.accounts.get(st.session_state.current_user)
        if not account:
            st.sidebar.error("Account profile not found.")
        elif not verify_password(current_password, str(account.get("password_hash", ""))):
            st.sidebar.error("Current password is incorrect.")
        elif len(new_password.strip()) < 8:
            st.sidebar.error("New password must be at least 8 characters.")
        elif new_password != confirm_password:
            st.sidebar.error("New password and confirmation do not match.")
        else:
            st.session_state.accounts[st.session_state.current_user]["password_hash"] = hash_password(new_password.strip())
            emit_audit_event(logger, "password_changed", {"user": st.session_state.current_user})
            st.sidebar.success("Password changed.")

    if st.sidebar.button("Log out", use_container_width=True):
        if st.session_state.current_user:
            save_listing_draft_for_user(st.session_state.current_user, st.session_state.active_listing_draft)
        st.session_state.authenticated = False
        st.session_state.current_user = ""
        st.rerun()


def render_messages_panel(logger: Any, current_user: str) -> None:
    messages = st.session_state.messages
    relevant = [m for m in messages if m["buyer"] == current_user or m["lister"] == current_user]

    st.markdown("---")
    st.header("Messages")
    if not relevant:
        st.info("No messages yet.")
        return

    for msg in reversed(relevant[-30:]):
        st.write(
            f"**{msg['timestamp']}** | Listing: {msg['listing_title']} | "
            f"Buyer: {msg['buyer']} | Lister: {msg['lister']}"
        )
        st.write(f"Message: {msg['message']}")
        st.caption(f"Preferred contact: {msg['contact_preference']}")


def render_update_assistant_panel(logger: Any) -> None:
    st.markdown("---")
    st.header("AI update assistant")
    st.caption("Checks for updates and always asks for approval before any update action.")
    st.write(f"Current app version: **{APP_VERSION}**")

    if st.button("Run update check"):
        result = check_for_available_update()
        st.session_state.update_check_result = result
        emit_audit_event(logger, "update_check_run", {"status": result.get("status", "unknown")})

    result = st.session_state.update_check_result
    status = str(result.get("status", "unknown"))
    message = str(result.get("message", "Update check not run yet."))

    st.write(f"Status: **{status}**")
    st.write(message)

    latest_commit = str(result.get("latest_commit", ""))
    deployed_commit = str(result.get("deployed_commit", "unknown"))
    if latest_commit:
        st.caption(f"Latest GitHub main commit: {latest_commit}")
    st.caption(f"Deployed commit (if configured): {deployed_commit}")

    if bool(result.get("update_needed", False)):
        st.warning("Update is recommended by policy. Approval is required before proceeding.")
        if st.button("Approve update notification"):
            emit_audit_event(logger, "update_approved", {"status": status, "latest_commit": latest_commit})
            st.info("Update approved. Push/redeploy the latest code to apply changes.")


def main() -> None:
    st.set_page_config(page_title="Online market listing intelligence", page_icon="🛍️")
    st.title("Online market listing intelligence")
    st.caption("Improve listing quality, detect fraud, recommend prices, and compute a trust score.")

    init_state()

    logger = configure_logging()
    rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

    try:
        expected_username, expected_password = get_auth_credentials(getattr(st, "secrets", None))
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    bootstrap_password = get_effective_password(expected_username, expected_password)
    ensure_default_account(expected_username, bootstrap_password)
    st.session_state.account_profiles.setdefault(expected_username, {"email": "", "phone": ""})

    try:
        pre_auth_client_id = get_client_identity(getattr(st.context, "headers", {}))
    except Exception:
        pre_auth_client_id = "local"

    if not st.session_state.authenticated:
        remembered = get_remembered_credentials(pre_auth_client_id)

        with st.form("login_form"):
            username = st.text_input("Username", value=remembered["username"])
            password = st.text_input("Password", type="password", value=remembered["password"])
            remember_credentials = st.checkbox(
                "Remember username and password on this device",
                value=bool(remembered["username"]),
            )
            submitted = st.form_submit_button("Log in")

        if submitted:
            try:
                if authenticate_account(username.strip(), password):
                    st.session_state.authenticated = True
                    st.session_state.current_user = username.strip()
                    st.session_state.pending_login_user = username.strip()
                    st.session_state.needs_upload_resume_choice = True
                    load_user_listing_draft(username.strip())

                    if remember_credentials:
                        save_remembered_credentials(pre_auth_client_id, username.strip(), password)
                    else:
                        clear_remembered_credentials(pre_auth_client_id)

                    emit_audit_event(logger, "login", {"status": "success", "user": username.strip()})
                    st.rerun()

                account = st.session_state.accounts.get(username.strip())
                if account and not is_contact_verified(account):
                    st.error("Account exists but is not verified yet. Complete verification to log in.")
                else:
                    st.error("Invalid username or password")
                emit_audit_event(logger, "login", {"status": "failed", "user": username.strip()})
            except Exception as exc:
                logger.exception("Login flow failed")
                st.error(f"Login failed unexpectedly: {exc}")

        with st.expander("Create account"):
            with st.form("create_account_form"):
                new_username = st.text_input("New username")
                new_password = st.text_input("New password", type="password")
                confirm_new_password = st.text_input("Confirm password", type="password")
                verify_method = st.selectbox("Verification method", ["Email", "Phone"])
                verify_value = st.text_input("Email or phone for verification")
                create_submit = st.form_submit_button("Create account and send verification code")

            if create_submit:
                clean_username = new_username.strip()
                clean_contact = verify_value.strip()

                if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", clean_username):
                    st.error("Username must be 3-32 chars and only include letters, numbers, _, ., or -")
                elif account_exists(clean_username):
                    st.error("That username already exists.")
                elif len(new_password.strip()) < 8:
                    st.error("Password must be at least 8 characters.")
                elif new_password != confirm_new_password:
                    st.error("Passwords do not match.")
                elif not clean_contact:
                    st.error("Email or phone is required for verification.")
                else:
                    email = clean_contact if verify_method == "Email" else ""
                    phone = clean_contact if verify_method == "Phone" else ""
                    create_account(clean_username, new_password.strip(), email=email, phone=phone)
                    st.session_state.account_profiles[clean_username] = {"email": email, "phone": phone}

                    code = build_verification_code()
                    st.session_state.pending_verifications[clean_username] = {
                        "purpose": "create_account",
                        "method": verify_method.lower(),
                        "value": clean_contact,
                        "code": code,
                        "expires_at": (datetime.now(timezone.utc).timestamp() + 600),
                    }
                    emit_audit_event(logger, "account_created", {"user": clean_username, "method": verify_method.lower()})
                    st.success("Account created. Enter your verification code below to activate login.")

                    if get_secret_value("DEMO_SHOW_VERIFICATION_CODES").lower() != "false":
                        st.info(f"Demo verification code for {clean_username}: {code}")

            with st.form("verify_new_account_form"):
                verify_username = st.text_input("Username to verify")
                verify_code = st.text_input("Verification code")
                verify_submit = st.form_submit_button("Verify account")

            if verify_submit:
                pending = st.session_state.pending_verifications.get(verify_username.strip())
                account = st.session_state.accounts.get(verify_username.strip())

                if not pending or pending.get("purpose") != "create_account":
                    st.error("No account verification is pending for this username.")
                elif datetime.now(timezone.utc).timestamp() > float(pending.get("expires_at", 0)):
                    st.error("Verification code expired. Create account again to generate a new code.")
                elif str(verify_code).strip() != str(pending.get("code", "")):
                    st.error("Invalid verification code.")
                elif not account:
                    st.error("Account not found.")
                else:
                    if pending.get("method") == "email":
                        account["verified_email"] = True
                    else:
                        account["verified_phone"] = True
                    st.session_state.pending_verifications.pop(verify_username.strip(), None)
                    emit_audit_event(logger, "account_verified", {"user": verify_username.strip()})
                    st.success("Account verified. You can now log in.")

        with st.expander("Forgot password?"):
            with st.form("forgot_password_request_form"):
                recover_username = st.text_input("Username for recovery")
                recovery_method = st.selectbox("Recovery method", ["Email", "Phone"])
                recovery_value = st.text_input("Recovery email or phone")
                recovery_new_password = st.text_input("New password", type="password")
                recovery_confirm_password = st.text_input("Confirm new password", type="password")
                recover_submit = st.form_submit_button("Send recovery verification code")

            if recover_submit:
                account = st.session_state.accounts.get(recover_username.strip())
                expected_value = ""
                if account:
                    expected_value = str(account.get("email", "")) if recovery_method == "Email" else str(account.get("phone", ""))

                if not account:
                    st.error("Account not found.")
                elif not expected_value:
                    st.error("No recovery contact is linked for this method.")
                elif recovery_value.strip().lower() != expected_value.strip().lower():
                    st.error("Recovery verification failed. Contact value does not match.")
                elif len(recovery_new_password.strip()) < 8:
                    st.error("New password must be at least 8 characters.")
                elif recovery_new_password != recovery_confirm_password:
                    st.error("New password and confirmation do not match.")
                else:
                    code = build_verification_code()
                    st.session_state.pending_verifications[recover_username.strip()] = {
                        "purpose": "reset_password",
                        "method": recovery_method.lower(),
                        "value": recovery_value.strip(),
                        "code": code,
                        "new_password_hash": hash_password(recovery_new_password.strip()),
                        "expires_at": (datetime.now(timezone.utc).timestamp() + 600),
                    }
                    emit_audit_event(
                        logger,
                        "password_reset_requested",
                        {"user": recover_username.strip(), "method": recovery_method.lower()},
                    )
                    st.success("Recovery code generated. Enter it below to complete password reset.")
                    if get_secret_value("DEMO_SHOW_VERIFICATION_CODES").lower() != "false":
                        st.info(f"Demo recovery code for {recover_username.strip()}: {code}")

            with st.form("forgot_password_verify_form"):
                verify_reset_username = st.text_input("Username for reset verification")
                verify_reset_code = st.text_input("Reset verification code")
                verify_reset_submit = st.form_submit_button("Verify and reset password")

            if verify_reset_submit:
                pending = st.session_state.pending_verifications.get(verify_reset_username.strip())
                account = st.session_state.accounts.get(verify_reset_username.strip())

                if not pending or pending.get("purpose") != "reset_password":
                    st.error("No password reset is pending for this username.")
                elif datetime.now(timezone.utc).timestamp() > float(pending.get("expires_at", 0)):
                    st.error("Recovery code expired. Request a new one.")
                elif str(verify_reset_code).strip() != str(pending.get("code", "")):
                    st.error("Invalid recovery code.")
                elif not account:
                    st.error("Account not found.")
                else:
                    account["password_hash"] = str(pending.get("new_password_hash", account.get("password_hash", "")))
                    st.session_state.pending_verifications.pop(verify_reset_username.strip(), None)
                    emit_audit_event(logger, "password_reset", {"user": verify_reset_username.strip()})
                    st.success("Password reset successful. Use your new password to log in.")
        st.stop()

    try:
        client_id = get_client_identity(getattr(st.context, "headers", {}))
        if not rate_limiter.allow_request(client_id):
            emit_audit_event(logger, "rate_limited", {"client": client_id})
            st.error("Too many requests from this client. Please wait a minute and try again.")
            st.stop()
    except Exception as exc:
        logger.exception("Client identification failed")
        st.error(f"Request setup failed unexpectedly: {exc}")
        st.stop()

    render_account_security_panel(logger=logger)
    render_update_assistant_panel(logger=logger)

    role = st.radio("Select your role", ["Lister", "Buyer"], horizontal=True)

    if role == "Buyer":
        listings = st.session_state.analyzed_listings
        st.header("Marketplace")
        if not listings:
            st.info("No listings available yet. Ask a lister to publish a listing first.")
        else:
            for key, item in reversed(list(listings.items())):
                with st.container(border=True):
                    st.subheader(f"{item['title']} - ${item['ai']['suggested_price']}")
                    st.caption(f"Location: {item['location']} | Listed by: {item['lister']}")
                    st.write(item["ai"]["generated_description"])
                    st.write(f"Condition: {item['ai']['condition']} | Year: {item['ai']['year']}")

                    with st.form(f"interest_form_{key}"):
                        message = st.text_area("Message to lister", placeholder="Hi, I am interested. Is this still available?")
                        contact_preference = st.selectbox("Preferred contact", ["In-platform messages", "Email", "Phone"])
                        send_interest = st.form_submit_button("I'm interested")

                    if send_interest:
                        clean_message = message.strip()
                        if not clean_message:
                            st.error("Please enter a message before sending.")
                        else:
                            st.session_state.messages.append(
                                {
                                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                    "listing_key": key,
                                    "listing_title": item["title"],
                                    "buyer": st.session_state.current_user,
                                    "lister": item["lister"],
                                    "message": clean_message,
                                    "contact_preference": contact_preference,
                                }
                            )
                            emit_audit_event(
                                logger,
                                "buyer_interest_sent",
                                {
                                    "listing_key": key,
                                    "buyer": st.session_state.current_user,
                                    "lister": item["lister"],
                                },
                            )
                            st.success("Interest message sent to lister.")

        render_messages_panel(logger=logger, current_user=st.session_state.current_user)
        render_reviews_section(logger=logger, client_id=client_id)
        return

    listing_field_errors: dict[str, str] = st.session_state.listing_field_errors
    render_listing_field_highlights(listing_field_errors)

    vision_enabled = bool(get_secret_value("OPENAI_API_KEY"))
    if vision_enabled:
        st.caption("AI vision model is enabled for uploaded photo analysis.")
    else:
        st.caption("AI vision model is not configured. Using heuristic photo analysis.")

    use_location_detect = st.checkbox("Allow location suggestion from my connection (permission-based)")
    if st.button("Suggest my current location", disabled=not use_location_detect):
        detected = suggest_location_from_ip(client_id)
        if detected:
            st.session_state.detected_location = detected
            st.success(f"Suggested location: {detected}")
        else:
            st.warning("Could not determine your location automatically. Enter location manually.")

    default_location = st.session_state.detected_location if st.session_state.detected_location else "Austin, TX"

    if st.session_state.needs_upload_resume_choice and st.session_state.pending_login_user == st.session_state.current_user:
        existing_draft = st.session_state.listing_drafts.get(st.session_state.current_user, get_default_listing_draft(default_location))
        has_existing_content = bool(
            str(existing_draft.get("title", "")).strip()
            or str(existing_draft.get("description", "")).strip()
            or str(existing_draft.get("location", "")).strip()
            or int(existing_draft.get("price", 699)) != 699
            or bool(existing_draft.get("uploaded_photos", []))
        )

        if has_existing_content:
            st.info("You have an unfinished upload draft from a previous session.")
            choice = st.radio(
                "Upload draft option",
                ["Resume where I left off", "Start again"],
                horizontal=True,
                key="resume_upload_choice",
            )
            if st.button("Apply draft choice", key="apply_upload_choice"):
                if choice == "Start again":
                    clear_listing_draft_for_user(st.session_state.current_user, default_location)
                    emit_audit_event(
                        logger,
                        "draft_reset_on_login",
                        {"user": st.session_state.current_user},
                    )
                else:
                    load_user_listing_draft(st.session_state.current_user, default_location)
                    emit_audit_event(
                        logger,
                        "draft_resumed_on_login",
                        {"user": st.session_state.current_user},
                    )
                st.session_state.needs_upload_resume_choice = False
                st.session_state.pending_login_user = ""
                st.rerun()
            st.stop()

        st.session_state.needs_upload_resume_choice = False
        st.session_state.pending_login_user = ""

    active_draft = st.session_state.active_listing_draft
    if not str(active_draft.get("location", "")).strip():
        active_draft["location"] = default_location

    st.markdown("### Upload draft")
    col_draft_1, col_draft_2 = st.columns(2)
    with col_draft_1:
        if st.button("Save draft", use_container_width=True):
            save_listing_draft_for_user(st.session_state.current_user, active_draft)
            emit_audit_event(logger, "draft_saved", {"user": st.session_state.current_user})
            st.success("Draft saved. You can resume it after logging in again.")
    with col_draft_2:
        if st.button("Start new draft", use_container_width=True):
            clear_listing_draft_for_user(st.session_state.current_user, default_location)
            emit_audit_event(logger, "draft_cleared", {"user": st.session_state.current_user})
            st.success("Started a new draft.")
            st.rerun()

    with st.form("listing_form"):
        uploaded_photos = st.file_uploader(
            "Listing photos (upload front, back, and sideways views)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )

        if uploaded_photos:
            active_draft["uploaded_photos"] = [f.name for f in uploaded_photos]

        photo_views: list[str] = []
        for idx, uploaded_photo in enumerate(uploaded_photos or []):
            view_label = st.selectbox(
                f"Photo view for {uploaded_photo.name}",
                ["Front side", "Back side", "Sideways", "Other"],
                index=["Front side", "Back side", "Sideways", "Other"].index(
                    active_draft.get("photo_views", {}).get(uploaded_photo.name, "Other")
                    if active_draft.get("photo_views", {}).get(uploaded_photo.name, "Other") in ["Front side", "Back side", "Sideways", "Other"]
                    else "Other"
                ),
                key=f"photo_view_{idx}_{uploaded_photo.name}",
            )
            photo_views.append(view_label)
            active_draft.setdefault("photo_views", {})[uploaded_photo.name] = view_label

        title = st.text_input(
            "Listing title",
            value=str(active_draft.get("title", "")),
            placeholder="Used iPhone 13 Pro Max",
        )
        active_draft["title"] = title
        if "title" in listing_field_errors:
            st.error(listing_field_errors["title"])

        description = st.text_area(
            "Description",
            value=str(active_draft.get("description", "")),
            placeholder="Describe the item clearly and include condition details.",
        )
        active_draft["description"] = description
        if "description" in listing_field_errors:
            st.error(listing_field_errors["description"])

        price = st.number_input("Price", min_value=0, step=1, value=int(active_draft.get("price", 699)))
        active_draft["price"] = int(price)
        if "price" in listing_field_errors:
            st.error(listing_field_errors["price"])

        location = st.text_input(
            "Location",
            value=str(active_draft.get("location", default_location)),
            placeholder="Austin, TX",
        )
        active_draft["location"] = location
        if "location" in listing_field_errors:
            st.error(listing_field_errors["location"])

        submitted = st.form_submit_button("Analyze listing")

    if submitted:
        st.session_state.listing_field_errors = build_listing_field_errors(title, description, price, location)
        listing_field_errors = st.session_state.listing_field_errors

        if listing_field_errors:
            emit_audit_event(
                logger,
                "validation_failed",
                {
                    "client": client_id,
                    "field_errors": listing_field_errors,
                },
            )
            st.warning("Please fix the highlighted fields and submit again.")
            st.rerun()

        try:
            title, description, price, location = validate_listing_input(title=title, description=description, price=price, location=location)
            quality = analyze_listing_quality(title=title, description=description, price=price, location=location)
            fraud = detect_fraud_signals(title=title, description=description, price=price, location=location)
            price_recommendation = recommend_price(title=title, description=description, price=price, location=location)
            trust = compute_trust_score(title=title, description=description, price=price, location=location)
            photo_insights = []
            for idx, photo in enumerate(uploaded_photos or []):
                view_label = photo_views[idx] if idx < len(photo_views) else "Other"
                photo_insights.append(analyze_uploaded_photo(photo, view_label))

            ai_suggestion = generate_ai_suggestion_with_optional_vision(
                title=title,
                description=description,
                location=location,
                current_price=float(price),
                photo_insights=photo_insights,
                pipeline_price=price_recommendation["suggested_price"],
                uploaded_photos=uploaded_photos or [],
                photo_views=photo_views,
            )
            st.session_state.listing_field_errors = {}
            clear_listing_draft_for_user(st.session_state.current_user, default_location)
        except ValueError as exc:
            emit_audit_event(logger, "validation_failed", {"client": client_id, "error": str(exc)})
            st.error(f"Input validation failed: {exc}")
            st.stop()
        except Exception as exc:
            logger.exception("Listing analysis failed")
            emit_audit_event(logger, "analysis_failed", {"client": client_id, "error": str(exc)})
            st.error(f"Analysis failed unexpectedly: {exc}")
            st.stop()

        emit_audit_event(logger, "listing_analyzed", {"client": client_id, "score": quality["score"]})

        current_listing_key = build_listing_key(title, location, st.session_state.current_user)
        profile = st.session_state.account_profiles.get(st.session_state.current_user, {"email": "", "phone": ""})
        st.session_state.analyzed_listings[current_listing_key] = {
            "title": title,
            "location": location,
            "description": description,
            "price": float(price),
            "lister": st.session_state.current_user,
            "contact_email": profile.get("email", ""),
            "contact_phone": profile.get("phone", ""),
            "ai": ai_suggestion,
        }
        st.session_state.last_listing_key = current_listing_key

        st.subheader("Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Quality score", f"{quality['score']}/100")
        col2.metric("Fraud risk", f"{fraud['risk_score']}/100")
        col3.metric("Suggested price", f"${ai_suggestion['suggested_price']}")
        col4.metric("Trust score", f"{trust['trust_score']}/100")

        st.markdown("### Quality insights")
        for item in quality["recommendations"]:
            st.write(f"- {item}")

        st.markdown("### Fraud signals")
        for item in fraud["reasons"]:
            st.write(f"- {item}")

        st.markdown("### Price rationale")
        st.write(price_recommendation["rationale"])

        st.markdown("### AI-assisted product summary")
        st.write(f"Analysis source: **{ai_suggestion['source']}** ({ai_suggestion['model']})")
        st.write(f"Model confidence: **{ai_suggestion['confidence']}**")
        st.write(f"Product name: **{ai_suggestion['product_name']}**")
        st.write(f"Estimated production year: **{ai_suggestion['year']}**")
        st.write(f"Condition assessment: **{ai_suggestion['condition']}**")
        st.write(
            f"Market value range: **${ai_suggestion['market_low']} - ${ai_suggestion['market_high']}** "
            f"(recommended list price: **${ai_suggestion['suggested_price']}**)"
        )
        st.text_area("Generated detailed description", value=ai_suggestion["generated_description"], height=180)

        if ai_suggestion.get("issues_to_edit"):
            st.markdown("### AI edits to improve listing")
            for issue in ai_suggestion["issues_to_edit"]:
                st.write(f"- {issue}")

        st.markdown("### Listing photos")
        if not uploaded_photos:
            st.info("No photos uploaded. Adding clear photos can improve buyer confidence and conversion.")
        else:
            columns_per_row = 3
            for start in range(0, len(uploaded_photos), columns_per_row):
                row_files = uploaded_photos[start : start + columns_per_row]
                row_insights = photo_insights[start : start + columns_per_row]
                cols = st.columns(len(row_files))

                for idx, uploaded_photo in enumerate(row_files):
                    insight = row_insights[idx]
                    with cols[idx]:
                        st.image(uploaded_photo, caption=f"Photo {start + idx + 1}", use_container_width=True)
                        st.caption(
                            f"{insight['view']} | {insight['format']} | {insight['width']}x{insight['height']} | "
                            f"{insight['megapixels']} MP | {insight['file_size_kb']} KB"
                        )

    render_messages_panel(logger=logger, current_user=st.session_state.current_user)
    render_reviews_section(logger=logger, client_id=client_id)


if __name__ == "__main__":
    main()
