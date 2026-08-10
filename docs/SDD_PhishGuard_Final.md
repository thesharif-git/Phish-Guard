# Software Design Document (SDD)
## PhishGuard AI: Intelligent Phishing Detection Ecosystem

**Author:** Muhammad Sharif Hassan  
**Supervisor:** Dr. Mareena Rashid  
**Department:** Computer Science & IT  
**University:** The Islamia University of Bahawalpur  
**Date:** May 12, 2026  

---

## 1. Introduction

### 1.1 Purpose
The purpose of this Software Design Document (SDD) is to describe the architecture and system design of the **PhishGuard AI** platform. It provides a detailed breakdown of the system components, data structures, and user interface designs that fulfill the requirements outlined in the Software Requirements Specification (SRS).

### 1.2 Scope
This document covers the high-level architecture (backend and database), the component-level design of the core scanning modules (Email, URL, and Domain Reputation), and the UI/UX design system implemented for the frontend interfaces.

---

## 2. System Architecture

PhishGuard AI follows a modern, decoupled monolithic architecture using the **FastAPI** framework.

### 2.1 High-Level Architecture
*   **Presentation Layer (Frontend):** Server-side rendered HTML using **Jinja2** templates styled with **Tailwind CSS**. JavaScript is used for dynamic client-side interactions (e.g., triggering API calls and updating UI states).
*   **Application Layer (Backend):** Built with **FastAPI** (Python 3.10+). It handles routing, authentication (JWT/OAuth), and the core business logic, including the orchestration of ML models and external APIs.
*   **Data Layer (Database):** Utilizes **SQLModel** (built on SQLAlchemy and Pydantic) to interface with a SQLite/PostgreSQL database, ensuring type safety and streamlined ORM operations.

### 2.2 Directory Structure
The application is structured for scalability and separation of concerns:
*   `/core`: Centralized configuration, logging, and application setup.
*   `/data`: Database connection logic, SQLModel schema definitions (`models.py`), and data access objects (`store.py`).
*   `/routers`: API and Page route definitions.
*   `/services`: Core business logic, including the AI Scan Engine and Security utilities.
*   `/templates`: Jinja2 HTML templates for the frontend views.
*   `/static`: Static assets (CSS, JS, media).

---

## 3. Component Design: Core Scanning Modules

Since the initial SRS, PhishGuard AI has expanded from a single email scanner into a comprehensive multi-vector intelligence suite. 

### 3.1 Email Scan Module (`/scan`)
*   **Function:** Analyzes raw text or `.eml` files.
*   **UI Structure:** A 2-column layout focusing on input processing and displaying a unified "Scan Result" card with a Risk Score Gauge and extracted threat indicators.

### 3.2 URL Link Detector (`/url-scan`)
*   **Function:** Dedicated interface for quick URL safety checks without requiring full email headers.
*   **UI Structure:**
    *   **Top Row:** A streamlined URL input field alongside the primary "Scan Result" gauge.
    *   **Middle Row (3-Column Grid):** Breaks down the result into:
        1.  **URL Overview** (Domain, Status, Redirects, SSL Cert).
        2.  **Risk Breakdown** (Granular sub-scores using progress bars).
        3.  **AI Explanation & Recommended Action** (Human-readable intelligence).

### 3.3 Domain Reputation Checker (`/domain-reputation`)
*   **Function:** A deep-dive technical intelligence dashboard for analyzing the global trust and infrastructure of a domain.
*   **UI Structure:**
    *   **Summary Row:** Domain input and an overarching "Reputation Score" gauge (e.g., 0-100 scale).
    *   **Intelligence Grid (3-Column):**
        1.  **Whois Intelligence:** Registrar details, creation date, and expiry.
        2.  **DNS Health:** Verification of SPF, DMARC, DNSSEC, and SSL configurations.
        3.  **Security Status:** Global blacklist query results (e.g., Spamhaus, Safe Browsing).
    *   **Infrastructure Row:** Displays active subdomains, main IPs, and CDN details.

---

## 4. UI/UX Design System

The frontend has been refined to adhere to a strict, high-end, minimalist aesthetic suitable for enterprise security products.

### 4.1 Color Palette & Typography
*   **Typography:** Google Font **'Inter'** is used globally for high legibility.
*   **Theme:** "Slate & White" minimalism. Backgrounds use very light grays (`#f8fafc`), cards are crisp white, and text utilizes dark slates (`slate-900` for headings, `slate-500` for secondary text).
*   **Alert Colors:** 
    *   Red (`red-600` / `red-50`) for High Risk/Dangerous items.
    *   Emerald (`emerald-600` / `emerald-50`) for Safe/High Trust items.
    *   Amber (`amber-500`) for Suspicious items.

### 4.2 Component Library
*   **Navigation Sidebar:** A synchronized, persistent left-hand sidebar containing links to all core modules (Overview, Scan, URL Link Detector, Domain Reputation), ensuring seamless user flow.
*   **Risk Gauges:** Custom SVG-based circular progress indicators used uniformly across all scanning modules to display risk/trust scores clearly.
*   **Card Grids:** Content is organized into `bg-white rounded-xl shadow-sm border border-slate-200` cards. The layout strictly enforces CSS Grid alignment (e.g., equal-height 3-column rows) to prevent visual bloat and overlap.

---

## 5. Database Schema Design

The system utilizes the following core data models via SQLModel:

### 5.1 `User` Model
*   **id:** UUID (Primary Key)
*   **email:** String (Unique)
*   **hashed_password:** String
*   **auth_provider:** String (e.g., "local", "google")
*   **is_active:** Boolean

### 5.2 `ScanRecord` Model
*   **scan_id:** String (Primary Key)
*   **user_id:** UUID (Foreign Key)
*   **subject/domain:** String (The scanned entity)
*   **status:** String ("Clean", "Suspicious", "Phishing")
*   **risk_score:** Integer (0-100)
*   **full_analysis:** JSON/Dict (Stores granular ML scores, VirusTotal results, and heuristic flags)
*   **timestamp:** DateTime

### 5.3 `Campaign` Model (For automated aggregation)
*   **id:** String (Primary Key)
*   **name:** String (Auto-generated descriptor)
*   **threat_type:** String (e.g., "Credential Harvesting")
*   **target_brand:** String
*   **associated_scans:** List[ScanRecord]

---
*End of Document. Prepared for final academic review.*
