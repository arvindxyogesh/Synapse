"""Labeled prompt set for the semantic-cache benchmark.

Each cluster is one underlying question asked several different ways
(paraphrases) -- a second-or-later mention of a cluster should hit the
semantic cache. Each confuser is a *different* question that sits close in
topic to one cluster, so it's a realistic test of whether the cache
over-generalizes (a false positive) rather than an easy win.

Sized deliberately large (35 clusters, ~2 confusers each) rather than a
small handful: with too small a prompt universe, a long benchmark run
saturates into almost-all-exact-repeat traffic after a few hundred
requests, which trivially inflates precision/recall regardless of whether
the similarity threshold is doing anything -- see benchmark.py's
"novel_only" metric, which is what this size is meant to keep meaningful
for longer.
"""

CLUSTERS: list[dict] = [
    {
        "id": "reset_password",
        "paraphrases": [
            "How do I reset my password?",
            "I forgot my password, how can I reset it?",
            "What's the process for resetting a forgotten password?",
            "Can you walk me through resetting my account password?",
        ],
    },
    {
        "id": "rate_limits",
        "paraphrases": [
            "What are the API rate limits?",
            "How many requests per minute am I allowed to make?",
            "Is there a limit on how often I can call the API?",
            "What's the rate limit on this API?",
        ],
    },
    {
        "id": "auth_requests",
        "paraphrases": [
            "How do I authenticate my API requests?",
            "What's the auth method for calling the API?",
            "How do I add authentication to my requests?",
            "What header do I need to authenticate?",
        ],
    },
    {
        "id": "find_api_key",
        "paraphrases": [
            "Where can I find my API key?",
            "How do I get an API key?",
            "Where do I go to see my API key?",
            "I can't find my API key, where is it?",
        ],
    },
    {
        "id": "cancel_subscription",
        "paraphrases": [
            "How do I cancel my subscription?",
            "I want to cancel my plan, how do I do that?",
            "What's the process to cancel my subscription?",
            "How can I stop my subscription?",
        ],
    },
    {
        "id": "payment_methods",
        "paraphrases": [
            "What payment methods do you accept?",
            "Can I pay with PayPal?",
            "What forms of payment are supported?",
            "Do you accept credit cards?",
        ],
    },
    {
        "id": "export_data",
        "paraphrases": [
            "How do I export my data?",
            "Can I download all my data?",
            "What's the way to export my account data?",
            "How can I get a copy of my data?",
        ],
    },
    {
        "id": "free_tier",
        "paraphrases": [
            "Is there a free tier?",
            "Do you offer a free plan?",
            "Can I use this for free?",
            "Is there a free version available?",
        ],
    },
    {
        "id": "add_team_member",
        "paraphrases": [
            "How do I add a team member?",
            "How can I invite someone to my team?",
            "What's the process for adding a new team member?",
            "How do I invite a teammate?",
        ],
    },
    {
        "id": "staging_vs_prod",
        "paraphrases": [
            "What's the difference between staging and production environments?",
            "How is staging different from production?",
            "What separates the staging env from prod?",
            "Staging vs production, what's the difference?",
        ],
    },
    {
        "id": "webhooks_setup",
        "paraphrases": [
            "How do I set up webhooks?",
            "What's the process for configuring a webhook?",
            "How can I receive webhook events?",
            "How do I add a webhook endpoint?",
        ],
    },
    {
        "id": "429_error",
        "paraphrases": [
            "Why am I getting a 429 error?",
            "What does a 429 response mean?",
            "My requests are failing with status 429, why?",
            "I keep getting HTTP 429, what's wrong?",
        ],
    },
    {
        "id": "upgrade_plan",
        "paraphrases": [
            "How do I upgrade my plan?",
            "How can I move to a higher tier plan?",
            "What's the process to upgrade my subscription?",
            "How do I switch to a bigger plan?",
        ],
    },
    {
        "id": "python_support",
        "paraphrases": [
            "Can I use this with Python?",
            "Is there a Python SDK?",
            "Does this support Python?",
            "How do I call this from Python?",
        ],
    },
    {
        "id": "delete_account",
        "paraphrases": [
            "How do I delete my account?",
            "I want to permanently delete my account, how?",
            "What's the process for account deletion?",
            "How can I close my account for good?",
        ],
    },
    {
        "id": "enable_2fa",
        "paraphrases": [
            "How do I enable two-factor authentication?",
            "How can I turn on 2FA for my account?",
            "What's the process for setting up two-factor auth?",
            "How do I add 2FA to my login?",
        ],
    },
    {
        "id": "sso_setup",
        "paraphrases": [
            "How do I set up single sign-on?",
            "What's the process for configuring SSO?",
            "How can I enable SSO for my organization?",
            "How do I connect our identity provider for SSO?",
        ],
    },
    {
        "id": "saml_config",
        "paraphrases": [
            "How do I configure SAML?",
            "What's the process for setting up a SAML integration?",
            "How can I connect our SAML identity provider?",
            "How do I add SAML-based login?",
        ],
    },
    {
        "id": "custom_roles",
        "paraphrases": [
            "How do I create a custom role?",
            "Can I define custom permission roles?",
            "What's the process for setting up a custom role?",
            "How do I set up a role with specific permissions?",
        ],
    },
    {
        "id": "ip_allowlist",
        "paraphrases": [
            "How do I set up an IP allowlist?",
            "Can I restrict access to specific IP addresses?",
            "What's the process for allowlisting IPs?",
            "How do I limit access by IP address?",
        ],
    },
    {
        "id": "custom_domain",
        "paraphrases": [
            "How do I set up a custom domain?",
            "Can I use my own domain?",
            "What's the process for connecting a custom domain?",
            "How do I point my domain at this?",
        ],
    },
    {
        "id": "audit_logs",
        "paraphrases": [
            "How do I view audit logs?",
            "Where can I see the audit log for my account?",
            "What's the process for accessing audit history?",
            "How do I check who made a change?",
        ],
    },
    {
        "id": "data_backups",
        "paraphrases": [
            "How do I back up my data?",
            "Are backups taken automatically?",
            "What's the process for restoring from a backup?",
            "Can I download a backup of my account?",
        ],
    },
    {
        "id": "usage_alerts",
        "paraphrases": [
            "How do I set up usage alerts?",
            "Can I get notified when I'm close to my quota?",
            "What's the process for configuring usage alerts?",
            "How do I turn on quota notifications?",
        ],
    },
    {
        "id": "escalate_support",
        "paraphrases": [
            "How do I escalate a support ticket?",
            "What's the process for escalating an issue?",
            "How can I get a support ticket prioritized?",
            "How do I reach a human for urgent issues?",
        ],
    },
    {
        "id": "localization",
        "paraphrases": [
            "How do I change the display language?",
            "Can I use this in a language other than English?",
            "What's the process for switching locales?",
            "How do I set my preferred language?",
        ],
    },
    {
        "id": "mobile_app",
        "paraphrases": [
            "Is there a mobile app?",
            "Can I use this on my phone?",
            "Do you have an iOS or Android app?",
            "Is there a mobile version available?",
        ],
    },
    {
        "id": "api_versioning",
        "paraphrases": [
            "How does API versioning work?",
            "What's the process for pinning an API version?",
            "How do I know which API version I'm using?",
            "How do I request a specific API version?",
        ],
    },
    {
        "id": "deprecation_policy",
        "paraphrases": [
            "What's the deprecation policy for old API versions?",
            "How much notice do you give before deprecating a feature?",
            "What's the process when an endpoint gets deprecated?",
            "How do I find out about upcoming breaking changes?",
        ],
    },
    {
        "id": "data_privacy",
        "paraphrases": [
            "How is my data handled for GDPR compliance?",
            "What's your data privacy policy?",
            "How do I request deletion of my personal data?",
            "What's the process for a data subject access request?",
        ],
    },
    {
        "id": "uptime_sla",
        "paraphrases": [
            "What's your uptime SLA?",
            "What uptime guarantee do you offer?",
            "Where can I check your status page?",
            "What happens if you miss the uptime SLA?",
        ],
    },
    {
        "id": "bulk_operations",
        "paraphrases": [
            "How do I perform a bulk update?",
            "Can I update multiple records at once?",
            "What's the process for a bulk import?",
            "How do I batch multiple requests together?",
        ],
    },
    {
        "id": "idempotency_keys",
        "paraphrases": [
            "How do idempotency keys work?",
            "What's the process for making a request idempotent?",
            "How do I avoid duplicate charges on retry?",
            "How do I safely retry a failed request?",
        ],
    },
    {
        "id": "pagination",
        "paraphrases": [
            "How does pagination work in the API?",
            "What's the process for paging through results?",
            "How do I get the next page of results?",
            "How do I control the page size in a list request?",
        ],
    },
    {
        "id": "sharing_permissions",
        "paraphrases": [
            "How do I share a project with someone?",
            "What's the process for setting sharing permissions?",
            "How can I give someone view-only access?",
            "How do I create a shareable link?",
        ],
    },
]

# Each confuser sits near one cluster's topic but asks something genuinely
# different -- a cache that's too permissive will wrongly reuse a cached
# answer for these (a false positive). Most clusters have two confusers so
# the "hard" near-miss test doesn't run out after a few hundred requests.
CONFUSERS: list[dict] = [
    {"near": "reset_password", "prompt": "How do I change my email address?"},
    {"near": "reset_password", "prompt": "How do I change my username?"},
    {"near": "rate_limits", "prompt": "What happens to in-flight requests during a deploy?"},
    {"near": "rate_limits", "prompt": "Is there a limit on request body size?"},
    {"near": "auth_requests", "prompt": "How do I rotate my API key?"},
    {"near": "auth_requests", "prompt": "How do I revoke an API key?"},
    {"near": "find_api_key", "prompt": "How do I see my usage for the current month?"},
    {"near": "cancel_subscription", "prompt": "How do I pause my subscription instead of cancelling?"},
    {"near": "cancel_subscription", "prompt": "How do I downgrade to a cheaper plan?"},
    {"near": "payment_methods", "prompt": "How do I get a copy of an old invoice?"},
    {"near": "payment_methods", "prompt": "How do I update my billing address?"},
    {"near": "export_data", "prompt": "How do I import data from a CSV?"},
    {"near": "free_tier", "prompt": "What's included in the enterprise plan?"},
    {"near": "add_team_member", "prompt": "How do I remove a team member?"},
    {"near": "add_team_member", "prompt": "How do I change a team member's role?"},
    {"near": "staging_vs_prod", "prompt": "How do I roll back a production deploy?"},
    {"near": "webhooks_setup", "prompt": "How do I test a webhook without triggering the real event?"},
    {"near": "webhooks_setup", "prompt": "Why did my webhook stop receiving events?"},
    {"near": "429_error", "prompt": "Why am I getting a 500 error?"},
    {"near": "429_error", "prompt": "Why am I getting a 401 error?"},
    {"near": "upgrade_plan", "prompt": "How do I see a comparison of all the plans?"},
    {"near": "python_support", "prompt": "Is there a Node.js SDK?"},
    {"near": "python_support", "prompt": "Is there a Go SDK?"},
    {"near": "delete_account", "prompt": "How do I deactivate my account temporarily?"},
    {"near": "enable_2fa", "prompt": "How do I disable two-factor authentication?"},
    {"near": "enable_2fa", "prompt": "I lost my 2FA device, how do I recover access?"},
    {"near": "sso_setup", "prompt": "How do I set up OAuth for a third-party app?"},
    {"near": "saml_config", "prompt": "What SAML attributes do you require?"},
    {"near": "custom_roles", "prompt": "What are the default (built-in) roles?"},
    {"near": "ip_allowlist", "prompt": "How do I set up a VPN connection?"},
    {"near": "custom_domain", "prompt": "How do I set up email forwarding?"},
    {"near": "audit_logs", "prompt": "How long are logs retained for?"},
    {"near": "data_backups", "prompt": "How do I export a single project instead of everything?"},
    {"near": "usage_alerts", "prompt": "How do I see my current usage right now?"},
    {"near": "escalate_support", "prompt": "How do I file a feature request?"},
    {"near": "localization", "prompt": "How do I change my timezone?"},
    {"near": "mobile_app", "prompt": "Is there a browser extension?"},
    {"near": "api_versioning", "prompt": "Where can I see the full API changelog?"},
    {"near": "deprecation_policy", "prompt": "How do I subscribe to status page incident updates?"},
    {"near": "data_privacy", "prompt": "Are you SOC 2 compliant?"},
    {"near": "uptime_sla", "prompt": "How do I get a refund for downtime?"},
    {"near": "bulk_operations", "prompt": "How do I schedule a recurring job?"},
    {"near": "idempotency_keys", "prompt": "How long is an idempotency key valid for?"},
    {"near": "pagination", "prompt": "How do I sort results by a specific field?"},
    {"near": "pagination", "prompt": "How do I filter results by date range?"},
    {"near": "sharing_permissions", "prompt": "How do I see the version history of a document?"},
    {"near": "sharing_permissions", "prompt": "How do I add a comment to a shared document?"},
]
