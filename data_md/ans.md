```yaml
spring:
  cache:
    type: redis
  data:
    redis:
      host: ${REDIS_HOST:127.0.0.1}
      password: ${REDIS_PASSWORD:}
      port: ${REDIS_PORT:36379}
      database: ${REDIS_DATABASE:0}
  cloud:
    nacos:
      server-addr: localhost:8848
      username: nacos
      password: nacos
    sentinel:
      eager: true
      enabled: true
      transport:
        dashboard: localhost:8858
    openfeign:
      sentinel:
        enabled: true
      okhttp:
        enabled: true
      httpclient:
        enabled: false
      compression:
        request:
          enabled: true
        response:
          enabled: true
    # seata:
    #   enabled: true  # 启用开关
    #   application-id: ${spring.application.name}  # 应用ID
    #   tx-service-group: campus_tx_group  # 事务分组（必须）

# 暴露监控端点
management:
  endpoints:
    web:
      exposure:
        include: "*"  
  endpoint:
    health:
      show-details: ALWAYS

# mybaits-plus配置
mybatis-plus:
  mapper-locations: classpath:/mapper/*Mapper.xml
  global-config:
    banner: false
    db-config:
      id-type: auto
      table-underline: true
      logic-delete-value: 'now()'
      logic-not-delete-value: 'null'
  type-handlers-package: com.example.common.mybatis.handler
  configuration:
    map-underscore-to-camel-case: true
    shrink-whitespaces-in-sql: true

# 短信插件配置：https://www.yuque.com/vxixfq/pig/zw8udk 暂时冗余
sms:
  is-print: false # 是否打印日志
  config-type: yaml # 配置类型，yaml

application-dev.yaml
```

```yaml
spring:
  security:
    encode-key: 'admin123'
    ignore-clients:
      - test
      
campus-auth-dev.yaml
```

```yaml
spring:
  cloud:
    gateway:
      server:
        webflux:
          routes:
            # 认证中心
            - id: campus-auth
              uri: lb://campus-auth
              predicates:
                - Path=/auth/**
            
            # UPMS 模块
            - id: campus-upms-biz
              uri: lb://campus-upms-biz
              predicates:
                - Path=/admin/**
              filters:
                - name: RequestRateLimiter
                  args:
                    key-resolver: '#{@remoteAddrKeyResolver}'
                    redis-rate-limiter.replenishRate: 100
                    redis-rate-limiter.burstCapacity: 200

            # 固定路由转发配置
            - id: openapi
              uri: lb://campus-gateway
              predicates:
                - Path=/v3/api-docs/**
              filters:
                - RewritePath=/v3/api-docs/(?<path>.*), /$\{path}/$\{path}/v3/api-docs
campus-gateway-dev.yaml
```

```yaml
spring:
  datasource:
    type: com.zaxxer.hikari.HikariDataSource
    driver-class-name: com.mysql.cj.jdbc.Driver
    username: ${MYSQL_USERNAME:root}
    password: ${MYSQL_PASSWORD:root}
    url: jdbc:mysql://${MYSQL_HOST:127.0.0.1}:${MYSQL_PORT:3306}/${MYSQL_DB:pig}?characterEncoding=utf8&zeroDateTimeBehavior=convertToNull&useSSL=false&allowMultiQueries=true&useJDBCCompliantTimezoneShift=true&useLegacyDatetimeCode=false&serverTimezone=Asia/Shanghai&nullCatalogMeansCurrent=true&allowPublicKeyRetrieval=true

file:
  bucketName: s3demo
  local:
    enable: true
    base-path: /data/upload/img
campus-upms-biz-dev.yaml
```

