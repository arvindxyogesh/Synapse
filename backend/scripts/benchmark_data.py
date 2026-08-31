"""Labeled prompt set for the semantic-cache benchmark.

Each cluster is one underlying question asked several different ways
(paraphrases) -- a second-or-later mention of a cluster should hit the
semantic cache. Each confuser is a *different* question that sits close in
topic to one cluster, so it's a realistic test of whether the cache
over-generalizes (a false positive) rather than an easy win.
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
]

# Each confuser sits near one cluster's topic but asks something genuinely
# different -- a cache that's too permissive will wrongly reuse a cached
# answer for these (a false positive).
CONFUSERS: list[dict] = [
    {"near": "reset_password", "prompt": "How do I change my email address?"},
    {"near": "rate_limits", "prompt": "What happens to in-flight requests during a deploy?"},
    {"near": "auth_requests", "prompt": "How do I rotate my API key?"},
    {"near": "find_api_key", "prompt": "How do I see my usage for the current month?"},
    {"near": "cancel_subscription", "prompt": "How do I pause my subscription instead of cancelling?"},
    {"near": "payment_methods", "prompt": "How do I get a copy of an old invoice?"},
    {"near": "export_data", "prompt": "How do I import data from a CSV?"},
    {"near": "free_tier", "prompt": "What's included in the enterprise plan?"},
    {"near": "add_team_member", "prompt": "How do I remove a team member?"},
    {"near": "staging_vs_prod", "prompt": "How do I roll back a production deploy?"},
    {"near": "webhooks_setup", "prompt": "How do I test a webhook without triggering the real event?"},
    {"near": "429_error", "prompt": "Why am I getting a 500 error?"},
    {"near": "upgrade_plan", "prompt": "How do I downgrade to a cheaper plan?"},
    {"near": "python_support", "prompt": "Is there a Node.js SDK?"},
    {"near": "delete_account", "prompt": "How do I deactivate my account temporarily?"},
]
