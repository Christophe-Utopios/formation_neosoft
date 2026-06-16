"""
Corpus partagé entre les démos M4.

Mini-base documentaire de support technique d'une plateforme SaaS fictive,
pour avoir un cas d'usage parlant en démo (50 chunks).
"""
from __future__ import annotations

CHUNKS: list[dict] = [
    # Authentication
    {"id": 1, "content": "Pour configurer SAML 2.0 sur NovaCloud version 3.2, allez dans Paramètres > Authentification > SAML. Renseignez l'URL du fournisseur d'identité (IdP), le certificat X.509 et l'ID de l'entité.", "product": "NovaCloud", "version": "3.2", "category": "auth", "title": "Configuration SAML 2.0"},
    {"id": 2, "content": "L'authentification LDAP nécessite un compte de service avec lecture sur l'annuaire. Format du Bind DN : CN=svc_account,OU=Services,DC=domain,DC=fr. Port standard : 389, ou 636 pour LDAPS.", "product": "NovaCloud", "version": "3.0", "category": "auth", "title": "Configuration LDAP"},
    {"id": 3, "content": "Pour activer l'authentification multi-facteurs (MFA), chaque utilisateur doit configurer un authenticator (Google Authenticator, Microsoft Authenticator, ou clé FIDO2). Activation forcée via Paramètres > Sécurité > MFA obligatoire.", "product": "NovaCloud", "version": "3.2", "category": "auth", "title": "MFA et 2FA"},
    {"id": 4, "content": "OAuth 2.0 avec authorization code flow : créer une application dans Paramètres > Intégrations > OAuth. Récupérer client_id et client_secret. Rediriger vers /oauth/authorize avec scope demandé.", "product": "NovaCloud", "version": "3.1", "category": "auth", "title": "OAuth 2.0"},
    {"id": 5, "content": "Single Sign-On (SSO) supporte SAML 2.0, OAuth 2.0 et OpenID Connect (OIDC). Recommandation 2026 : OIDC pour les nouveaux déploiements, plus simple et standard.", "product": "NovaCloud", "version": "3.2", "category": "auth", "title": "SSO et fournisseurs"},

    # Errors
    {"id": 6, "content": "Erreur 401 Unauthorized : le token JWT est expiré ou invalide. Vérifier la durée de validité (15 min par défaut) et utiliser le refresh token pour obtenir un nouveau access token.", "product": "NovaCloud", "version": "3.2", "category": "errors", "title": "Erreur 401"},
    {"id": 7, "content": "Erreur 403 Forbidden : l'utilisateur est authentifié mais n'a pas les droits sur la ressource. Vérifier les rôles attribués via Paramètres > Utilisateurs > Rôles et permissions.", "product": "NovaCloud", "version": "3.0", "category": "errors", "title": "Erreur 403"},
    {"id": 8, "content": "Erreur 503 Service Unavailable sur l'endpoint /api/reports : indique que le service de génération de rapports est en surcharge. Réessayer avec backoff exponentiel après 5, 15, 60 secondes.", "product": "NovaCloud", "version": "3.2", "category": "errors", "title": "Erreur 503 reports"},
    {"id": 9, "content": "Erreur 429 Too Many Requests : limite de débit atteinte. La limite par défaut est de 100 requêtes/minute par utilisateur. Header Retry-After indique le délai avant retry.", "product": "NovaCloud", "version": "3.1", "category": "errors", "title": "Erreur 429"},
    {"id": 10, "content": "Erreur 502 Bad Gateway : problème temporaire avec le reverse proxy. Si persistant plus de 5 minutes, contacter le support avec l'X-Request-Id présent dans les headers de la réponse.", "product": "NovaCloud", "version": "3.2", "category": "errors", "title": "Erreur 502"},

    # API
    {"id": 11, "content": "L'API REST de NovaCloud est documentée à api.novacloud.io/docs. Format JSON. Authentification par Bearer token dans le header Authorization.", "product": "NovaCloud", "version": "3.2", "category": "api", "title": "API REST"},
    {"id": 12, "content": "Pour exporter des données en bulk via l'API, utiliser POST /api/v3/exports avec format=csv|json|parquet. La réponse contient une URL S3 valable 1 heure.", "product": "NovaCloud", "version": "3.2", "category": "api", "title": "Export bulk API"},
    {"id": 13, "content": "Webhooks : configurer une URL HTTPS dans Paramètres > Intégrations > Webhooks. Événements supportés : user.created, document.updated, alert.triggered. Signature HMAC SHA-256 dans X-Signature.", "product": "NovaCloud", "version": "3.1", "category": "api", "title": "Webhooks"},
    {"id": 14, "content": "Pagination des listes API : utiliser les paramètres page et page_size (max 100). Le header X-Total-Count indique le nombre total. Recommandation : utiliser cursor-based pagination via le paramètre after pour les gros volumes.", "product": "NovaCloud", "version": "3.0", "category": "api", "title": "Pagination API"},
    {"id": 15, "content": "Rate limiting de l'API : 100 req/min par token, 1000 req/min par organisation. Pour des besoins supérieurs, contacter votre Customer Success Manager pour augmentation du quota.", "product": "NovaCloud", "version": "3.2", "category": "api", "title": "Rate limit API"},

    # Reports
    {"id": 16, "content": "Création d'un rapport hebdomadaire automatique : Rapports > Nouveau > Modèle hebdomadaire. Définir les destinataires, le jour d'envoi (lundi 8h par défaut) et les filtres.", "product": "NovaCloud", "version": "3.2", "category": "reports", "title": "Rapport hebdo"},
    {"id": 17, "content": "Les rapports peuvent être exportés en PDF, XLSX ou CSV. Le PDF inclut les graphiques avec watermark de l'organisation. Le XLSX préserve les formules pour analyse offline.", "product": "NovaCloud", "version": "3.1", "category": "reports", "title": "Formats export rapports"},
    {"id": 18, "content": "Pour partager un rapport avec un utilisateur externe, générer un lien public avec date d'expiration. Mot de passe optionnel mais recommandé pour les données sensibles.", "product": "NovaCloud", "version": "3.2", "category": "reports", "title": "Partage rapports"},
    {"id": 19, "content": "Le rapport hebdomadaire n'est pas envoyé : vérifier d'abord les destinataires actifs, puis les logs d'envoi (Paramètres > Logs > Email). Les bounces sont retentés 3 fois sur 24h.", "product": "NovaCloud", "version": "3.2", "category": "reports", "title": "Dépannage rapports"},
    {"id": 20, "content": "Les rapports utilisent les fuseaux horaires de l'organisation. Configurer dans Paramètres > Organisation > Fuseau horaire. Par défaut : Europe/Paris.", "product": "NovaCloud", "version": "3.0", "category": "reports", "title": "Fuseau horaire rapports"},

    # Billing
    {"id": 21, "content": "La facturation est mensuelle au prorata des utilisateurs actifs. Un utilisateur est compté actif s'il s'est connecté au moins une fois dans le mois. Désactiver les comptes inactifs pour optimiser le coût.", "product": "NovaCloud", "version": "3.2", "category": "billing", "title": "Modèle facturation"},
    {"id": 22, "content": "Modes de paiement acceptés : carte bancaire (Visa, Mastercard, Amex), prélèvement SEPA, virement bancaire (uniquement pour Enterprise plans).", "product": "NovaCloud", "version": "3.2", "category": "billing", "title": "Modes paiement"},
    {"id": 23, "content": "Récupérer ses factures : Paramètres > Facturation > Historique. Téléchargement PDF disponible. Les factures sont aussi envoyées par email à l'adresse de facturation déclarée.", "product": "NovaCloud", "version": "3.1", "category": "billing", "title": "Récup factures"},
    {"id": 24, "content": "Changement de plan : Paramètres > Facturation > Changer de plan. Upgrade : effet immédiat, prorata. Downgrade : effet à la prochaine date de facturation.", "product": "NovaCloud", "version": "3.2", "category": "billing", "title": "Changement plan"},
    {"id": 25, "content": "Le code TVA intracommunautaire de votre organisation peut être renseigné dans Paramètres > Organisation > Informations fiscales. Validation automatique via VIES.", "product": "NovaCloud", "version": "3.0", "category": "billing", "title": "TVA intracommunautaire"},

    # Migration
    {"id": 26, "content": "Migration depuis ancienne version 2.x vers 3.x : utiliser l'outil novacloud-migrate disponible dans le portail support. Durée : ~30 min pour 1 To de données.", "product": "NovaCloud", "version": "3.0", "category": "migration", "title": "Migration 2.x vers 3.x"},
    {"id": 27, "content": "La compatibilité ascendante est garantie pour les API REST entre 3.0 et 3.x. Les API GraphQL ont fait l'objet de breaking changes en 3.2 (champ deprecated retiré).", "product": "NovaCloud", "version": "3.2", "category": "migration", "title": "Compatibilité API"},
    {"id": 28, "content": "Migration LDAP vers SAML : possible sans interruption en activant les deux modes simultanément pendant 30 jours. Ensuite, désactiver LDAP via Paramètres > Authentification.", "product": "NovaCloud", "version": "3.1", "category": "migration", "title": "Migration LDAP→SAML"},

    # Security
    {"id": 29, "content": "Conformité RGPD : NovaCloud est certifié ISO 27001 et hébergé en France (datacenter OVH Roubaix). Les données restent dans l'UE. DPA signable depuis Paramètres > Conformité.", "product": "NovaCloud", "version": "3.2", "category": "security", "title": "RGPD"},
    {"id": 30, "content": "Chiffrement des données : AES-256 au repos (S3 SSE-KMS), TLS 1.3 en transit. Possibilité de Customer-Managed Keys (CMK) sur l'offre Enterprise.", "product": "NovaCloud", "version": "3.2", "category": "security", "title": "Chiffrement"},
    {"id": 31, "content": "Logs d'audit : tous les accès et modifications sont enregistrés et conservés 12 mois (24 mois sur Enterprise). Export possible via API ou interface.", "product": "NovaCloud", "version": "3.0", "category": "security", "title": "Audit logs"},
    {"id": 32, "content": "Sauvegarde et restauration : sauvegardes quotidiennes incrémentales + hebdo complète. Rétention 30 jours (90 jours Enterprise). Restauration point-in-time disponible sur demande au support.", "product": "NovaCloud", "version": "3.1", "category": "security", "title": "Backups"},

    # Performance
    {"id": 33, "content": "Optimiser les performances : utiliser les index sur les champs filtrés fréquemment, activer le cache CDN (CloudFront) pour les ressources statiques, paginer les listes au-delà de 50 items.", "product": "NovaCloud", "version": "3.2", "category": "performance", "title": "Best practices perfs"},
    {"id": 34, "content": "Limites par défaut : 100 utilisateurs par organisation (Standard), 1000 (Pro), illimité (Enterprise). 10 Go stockage par utilisateur, 100 Go pour les fichiers partagés.", "product": "NovaCloud", "version": "3.2", "category": "performance", "title": "Limites & quotas"},
    {"id": 35, "content": "Support des grandes listes : utiliser cursor-based pagination via le paramètre after au lieu de la pagination offset. Les performances dégradent au-delà de 10 000 lignes en offset.", "product": "NovaCloud", "version": "3.0", "category": "performance", "title": "Pagination perfs"},

    # Mobile
    {"id": 36, "content": "Application mobile iOS (iOS 16+) et Android (Android 12+) disponible sur les stores officiels. Synchronisation offline : 7 derniers jours de données accessibles sans connexion.", "product": "NovaCloud Mobile", "version": "2.5", "category": "mobile", "title": "Apps mobiles"},
    {"id": 37, "content": "Notifications push : configurer dans l'app mobile > Paramètres > Notifications. Granularité par type d'événement (alertes, mentions, rapports). Paramètre stocké côté utilisateur.", "product": "NovaCloud Mobile", "version": "2.5", "category": "mobile", "title": "Notifications push"},

    # Integrations
    {"id": 38, "content": "Intégration Slack : aller dans Paramètres > Intégrations > Slack. Cliquer sur Add to Slack, autoriser l'app. Choisir les canaux destinataires des alertes.", "product": "NovaCloud", "version": "3.2", "category": "integrations", "title": "Slack"},
    {"id": 39, "content": "Intégration Microsoft Teams : passer par Microsoft AppSource ou ajout manuel via webhook entrant. Activer la commande slash /novacloud pour interroger directement depuis Teams.", "product": "NovaCloud", "version": "3.1", "category": "integrations", "title": "MS Teams"},
    {"id": 40, "content": "Intégration Zapier et Make (ex-Integromat) : NovaCloud expose 50+ déclencheurs et actions. Catalogue complet sur zapier.com/apps/novacloud.", "product": "NovaCloud", "version": "3.0", "category": "integrations", "title": "Zapier"},

    # CLI
    {"id": 41, "content": "CLI novacloud-cli : installation via brew (Mac), apt (Debian/Ubuntu), ou téléchargement direct des binaires. Authentification : novacloud-cli login. Documentation : novacloud-cli help.", "product": "NovaCloud CLI", "version": "1.5", "category": "cli", "title": "CLI install"},
    {"id": 42, "content": "Commandes CLI courantes : novacloud-cli users list, novacloud-cli reports export, novacloud-cli backup create. Mode --json pour traitement par scripts.", "product": "NovaCloud CLI", "version": "1.5", "category": "cli", "title": "CLI commandes"},

    # Versioning
    {"id": 43, "content": "Politique de versioning : les versions 3.x sont supportées 18 mois après leur release. La version 3.0 sortie en 2024 sera EOL en juin 2026. Migration vers 3.2 recommandée.", "product": "NovaCloud", "version": "3.2", "category": "versioning", "title": "Politique versions"},
    {"id": 44, "content": "Channels de release : stable (production, mises à jour mensuelles), beta (mises à jour hebdo, opt-in), edge (daily builds, dev only).", "product": "NovaCloud", "version": "3.2", "category": "versioning", "title": "Release channels"},

    # Monitoring
    {"id": 45, "content": "Status page : status.novacloud.io affiche la santé en temps réel des services. S'abonner aux notifications par email ou Slack.", "product": "NovaCloud", "version": "3.2", "category": "monitoring", "title": "Status page"},
    {"id": 46, "content": "Métriques exposées : Prometheus endpoint /metrics (Enterprise). Métriques clés : http_requests_total, request_duration_seconds, active_users.", "product": "NovaCloud", "version": "3.2", "category": "monitoring", "title": "Prometheus"},

    # Customization
    {"id": 47, "content": "Personnalisation thème : Paramètres > Apparence > Thème. Couleur principale, logo, favicon. Mode sombre / clair / auto disponible.", "product": "NovaCloud", "version": "3.1", "category": "customization", "title": "Thème"},
    {"id": 48, "content": "Workflows personnalisés : créer dans Paramètres > Workflows. Déclencheurs : événements (création, modification), schedule (cron). Actions : notifications, appels API, scripts.", "product": "NovaCloud", "version": "3.2", "category": "customization", "title": "Workflows"},

    # Misc
    {"id": 49, "content": "Pour supprimer définitivement un compte utilisateur : Paramètres > Utilisateurs > Sélection > Supprimer. Données effacées sous 30 jours (RGPD droit à l'effacement).", "product": "NovaCloud", "version": "3.2", "category": "misc", "title": "Suppression compte"},
    {"id": 50, "content": "Export complet des données d'un utilisateur (RGPD article 20 - portabilité) : Paramètres > Confidentialité > Exporter mes données. Archive ZIP envoyée par email sous 24h.", "product": "NovaCloud", "version": "3.0", "category": "misc", "title": "Export RGPD"},
]


def get_corpus() -> list[dict]:
    return CHUNKS
