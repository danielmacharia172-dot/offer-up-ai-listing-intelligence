import re
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
    authenticate_user,
    get_auth_credentials,
    get_client_identity,
    validate_listing_input,
)


def init_state() -> None:
    defaults = {
        "authenticated": False,
        "current_user": "",
        "listing_field_errors": {},
        "buyer_reviews_by_listing": {},
        "analyzed_listings": {},
        "last_listing_key": None,
        "account_profiles": {},
        "password_overrides": {},
        "messages": [],
        "detected_location": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
    }


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


def render_account_security_panel(expected_username: str, expected_password: str, logger: Any) -> None:
    st.sidebar.header("Account and Security")
    st.sidebar.write(f"Signed in as: **{st.session_state.current_user}**")

    profiles = st.session_state.account_profiles
    profile = profiles.get(st.session_state.current_user, {"email": "", "phone": ""})

    with st.sidebar.form("link_contact_form"):
        email = st.text_input("Linked email", value=profile.get("email", ""), placeholder="name@example.com")
        phone = st.text_input("Linked phone", value=profile.get("phone", ""), placeholder="+1 555 123 4567")
        save_contact = st.form_submit_button("Save recovery contacts")

    if save_contact:
        profiles[st.session_state.current_user] = {"email": email.strip(), "phone": phone.strip()}
        emit_audit_event(logger, "backup_contact_updated", {"user": st.session_state.current_user})
        st.sidebar.success("Recovery contacts saved.")

    with st.sidebar.form("change_password_form"):
        current_password = st.text_input("Current password", type="password")
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        do_change = st.form_submit_button("Change password")

    if do_change:
        effective_password = get_effective_password(expected_username, expected_password)
        if st.session_state.current_user != expected_username:
            st.sidebar.error("Only the configured account can change password in this version.")
        elif not authenticate_user(expected_username, current_password, expected_username, effective_password):
            st.sidebar.error("Current password is incorrect.")
        elif len(new_password.strip()) < 8:
            st.sidebar.error("New password must be at least 8 characters.")
        elif new_password != confirm_password:
            st.sidebar.error("New password and confirmation do not match.")
        else:
            st.session_state.password_overrides[expected_username] = new_password.strip()
            emit_audit_event(logger, "password_changed", {"user": expected_username})
            st.sidebar.success("Password changed.")

    if st.sidebar.button("Log out", use_container_width=True):
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

    if not st.session_state.authenticated:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            try:
                effective_password = get_effective_password(expected_username, expected_password)
                if authenticate_user(username, password, expected_username, effective_password):
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    emit_audit_event(logger, "login", {"status": "success", "user": username})
                    st.rerun()
                emit_audit_event(logger, "login", {"status": "failed", "user": username})
                st.error("Invalid username or password")
            except Exception as exc:
                logger.exception("Login flow failed")
                st.error(f"Login failed unexpectedly: {exc}")

        with st.expander("Forgot password?"):
            with st.form("forgot_password_form"):
                recover_username = st.text_input("Username for recovery")
                recovery_method = st.selectbox("Recovery method", ["Email", "Phone"])
                recovery_value = st.text_input("Recovery email or phone")
                new_password = st.text_input("New password", type="password")
                confirm_password = st.text_input("Confirm new password", type="password")
                recover_submit = st.form_submit_button("Verify and reset password")

            if recover_submit:
                profile = st.session_state.account_profiles.get(recover_username, {"email": "", "phone": ""})
                expected_value = profile.get("email", "") if recovery_method == "Email" else profile.get("phone", "")

                if recover_username != expected_username:
                    st.error("Recovery is only available for the configured account.")
                elif not expected_value:
                    st.error("No recovery contact is linked yet. Log in and add email/phone first.")
                elif recovery_value.strip().lower() != expected_value.strip().lower():
                    st.error("Recovery verification failed. Contact value does not match.")
                elif len(new_password.strip()) < 8:
                    st.error("New password must be at least 8 characters.")
                elif new_password != confirm_password:
                    st.error("New password and confirmation do not match.")
                else:
                    st.session_state.password_overrides[expected_username] = new_password.strip()
                    emit_audit_event(logger, "password_reset", {"user": expected_username, "method": recovery_method.lower()})
                    st.success("Password reset successful. Use the new password to log in.")
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

    render_account_security_panel(expected_username=expected_username, expected_password=expected_password, logger=logger)

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

    use_location_detect = st.checkbox("Allow location suggestion from my connection (permission-based)")
    if st.button("Suggest my current location", disabled=not use_location_detect):
        detected = suggest_location_from_ip(client_id)
        if detected:
            st.session_state.detected_location = detected
            st.success(f"Suggested location: {detected}")
        else:
            st.warning("Could not determine your location automatically. Enter location manually.")

    default_location = st.session_state.detected_location if st.session_state.detected_location else "Austin, TX"

    with st.form("listing_form"):
        uploaded_photos = st.file_uploader(
            "Listing photos (upload front, back, and sideways views)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )

        photo_views: list[str] = []
        for idx, uploaded_photo in enumerate(uploaded_photos or []):
            view_label = st.selectbox(
                f"Photo view for {uploaded_photo.name}",
                ["Front side", "Back side", "Sideways", "Other"],
                key=f"photo_view_{idx}_{uploaded_photo.name}",
            )
            photo_views.append(view_label)

        title = st.text_input("Listing title", placeholder="Used iPhone 13 Pro Max")
        if "title" in listing_field_errors:
            st.error(listing_field_errors["title"])

        description = st.text_area("Description", placeholder="Describe the item clearly and include condition details.")
        if "description" in listing_field_errors:
            st.error(listing_field_errors["description"])

        price = st.number_input("Price", min_value=0, step=1, value=699)
        if "price" in listing_field_errors:
            st.error(listing_field_errors["price"])

        location = st.text_input("Location", value=default_location, placeholder="Austin, TX")
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

            ai_suggestion = generate_ai_suggestion(
                title=title,
                description=description,
                location=location,
                current_price=float(price),
                photo_insights=photo_insights,
                pipeline_price=price_recommendation["suggested_price"],
            )
            st.session_state.listing_field_errors = {}
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
        st.write(f"Product name: **{ai_suggestion['product_name']}**")
        st.write(f"Estimated production year: **{ai_suggestion['year']}**")
        st.write(f"Condition assessment: **{ai_suggestion['condition']}**")
        st.write(
            f"Market value range: **${ai_suggestion['market_low']} - ${ai_suggestion['market_high']}** "
            f"(recommended list price: **${ai_suggestion['suggested_price']}**)"
        )
        st.text_area("Generated detailed description", value=ai_suggestion["generated_description"], height=180)

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
