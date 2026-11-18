# 24 · Authentication and permissions

**You will learn:** MLflow's built-in basic-auth app, its permission model, and how
clients authenticate.

## Turning it on

```bash
pip install 'mlflow[auth]'
export MLFLOW_AUTH_CONFIG_PATH=auth.ini            # see below
export MLFLOW_FLASK_SERVER_SECRET_KEY=<random, same on every replica>
mlflow server --app-name basic-auth ...
```

```ini
[mlflow]
default_permission = READ
database_uri = sqlite:///basic_auth.db        ; use Postgres in production
admin_username = admin
admin_password = ...                         ; or MLFLOW_AUTH_ADMIN_PASSWORD. Only used on first start
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
```

## Permission model

| Permission | Read | Edit (log, update) | Delete | Manage (grant others) |
|---|---|---|---|---|
| `READ` | ✓ | | | |
| `EDIT` | ✓ | ✓ | | |
| `MANAGE` | ✓ | ✓ | ✓ | ✓ |
| `NO_PERMISSIONS` | | | | |

Permissions are granted per experiment, registered model, and other resources. A user who
creates a resource gets `MANAGE` on it. Everything else falls back to `default_permission`.
Admins can do everything. Roles group permissions for many users.

## Clients

```bash
export MLFLOW_TRACKING_USERNAME=alice
export MLFLOW_TRACKING_PASSWORD=...
# or MLFLOW_TRACKING_TOKEN for bearer tokens behind a proxy
```

Admin operations use `get_app_client("basic-auth", tracking_uri=...)`: `create_user`,
`grant_user_permission(username, "experiment", id, "EDIT")`, `update_user_admin`, `create_role`.

## Run it

`./book.sh run 24`. It starts its own temporary auth server on port 5102 and deletes it at
the end. Your book server is not changed.

## Beyond basic auth

For SSO, put MLflow behind an authenticating reverse proxy (oauth2-proxy, an ingress with
OIDC, or a cloud IAP), or plug in a custom `authorization_function`. Always use TLS. Basic
auth over plain HTTP sends passwords in clear text.

## Exercises

1. Give bob `EDIT` on the registered model `book-gated-classifier` only.
2. Create a `ci-bot` user for the chapter 15 workflow, with the minimum permissions it needs.
