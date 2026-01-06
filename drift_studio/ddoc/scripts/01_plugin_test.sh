# 도움말
ddoc --help
ddoc plugin --help

# 플러그인 목록/정보
ddoc plugin list
ddoc plugin info ddoc_builtins

# 플러그인 설치(예: 외부 패키지)
ddoc plugin install ddoc-plugin-nlp
ddoc plugin list
ddoc plugin info ddoc_nlp

# 우선순위/프로바이더 선택
ddoc drift --ref ref.txt --cur cur.txt --detector ks --provider ddoc_builtins
ddoc transform input.txt --transform text.upper --provider some_external_plugin