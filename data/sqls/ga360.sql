CREATE TEMP FUNCTION ActionToString (action STRING) AS (
  CASE action
    WHEN "1" THEN "Click through of product lists"
    WHEN "2" THEN "Product detail views"
    WHEN "3" THEN "Add product(s) to cart"
    WHEN "4" THEN "Remove product(s) from cart"
    WHEN "5" THEN "Check out"
    WHEN "6" THEN "Completed purchase"
    WHEN "7" THEN "Refund of purchase"
    WHEN "8" THEN "Checkout options"
  END
);

WITH
  base AS (
    SELECT
      *
    FROM
      `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
      UNNEST (hits) AS hits
    WHERE
      -- TODO: 1 month train, 1 month evaluation
      (
        _TABLE_SUFFIX BETWEEN '{start_date}' AND '{end_date}'
      )
      AND eCommerceAction.action_type != "0"
      AND geoNetwork.subContinent = "Northern America"
  ),
  duplicated AS (
    SELECT
      *
    FROM
      (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY
              fullVisitorId,
              round_sec,
              page.pageTitle
            ORDER BY
              TIME
          ) AS row_num
        FROM
          (
            SELECT
              *,
              CAST(ROUND(TIME / 5 / 1000) AS INT64) AS round_sec
            FROM
              base
          )
      )
    WHERE
      row_num = 1
  ),
  sequences AS (
    SELECT
      fullVisitorId,
      visitId,
      visitNumber,
      hitNumber AS hitNumber,
      TIME AS hitTime,
      round_sec,
      page.pageTitle AS pageTitle,
      page.pagePath AS pagePath,
      totals.totalTransactionRevenue / 1000000 AS transaction_USD,
      item,
      ARRAY (
        SELECT
          product.v2ProductName
        FROM
          UNNEST (duplicated.product) AS product
      ) AS productName,
      ARRAY (
        SELECT
          CONCAT(
            "$",
            CAST(product.ProductPrice / 1000000 AS STRING)
          )
        FROM
          UNNEST (duplicated.product) AS product
      ) AS productPrice,
      promotion eCommerceAction,
      ActionToString (eCommerceAction.action_type) AS actionType,
      geoNetwork.subContinent AS subContinent
    FROM
      duplicated
  ),
  sequence_nodes AS (
    SELECT
      fullVisitorId,
      visitID,
      hitTime,
      hitNumber,
      CASE
        WHEN actionType = "Click through of product lists" THEN TO_JSON_STRING(
          STRUCT (
            actionType AS actionType,
            -- TODO: add pagePath if nodes can increase
            NULL AS pageTitle,
            NULL AS pagePath,
            productName AS productName,
            productPrice AS productPrice
          )
        )
        WHEN actionType = "Product detail views" THEN TO_JSON_STRING(
          STRUCT (
            actionType AS actionType,
            NULL AS pageTitle,
            NULL AS pagePath,
            productName AS productName,
            productPrice AS productPrice
          )
        )
        WHEN actionType = "Add product(s) to cart" THEN TO_JSON_STRING(
          STRUCT (
            actionType AS actionType,
            NULL AS pageTitle,
            NULL AS pagePath,
            NULL AS productName,
            NULL AS productPrice
          )
        )
        WHEN actionType = "Remove product(s) from cart" THEN TO_JSON_STRING(
          STRUCT (
            actionType AS actionType,
            NULL AS pageTitle,
            NULL AS pagePath,
            NULL AS productName,
            NULL AS productPrice
          )
        )
        ELSE TO_JSON_STRING(
          STRUCT (
            actionType AS actionType,
            pageTitle AS pageTitle,
            NULL AS pagePath,
            NULL AS productName,
            NULL AS productPrice
          )
        )
      END AS event_detail
    FROM
      sequences
  ),
  padding_last_row AS (
    SELECT
      fullVisitorId,
      visitID,
      hitTime + 1 AS hitTime,
      hitNumber + 1 AS hitNumber,
      event_detail AS preceding_node,
      "session_end" AS succeeding_node,
    FROM
      (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY
              fullVisitorId,
              visitID
            ORDER BY
              hitNumber DESC
          ) AS row_num
        FROM
          sequence_nodes
      )
    WHERE
      row_num = 1
  ),
  pre_suc_nodes AS (
    SELECT
      fullVisitorId,
      visitID,
      hitTime,
      hitNumber,
      LAG(event_detail, 1, "session_start") OVER (
        PARTITION BY
          fullVisitorID,
          visitID
        ORDER BY
          hitNumber
      ) AS preceding_node,
      event_detail AS succeeding_node
    FROM
      sequence_nodes
  ),
  fully_pre_suc_node AS (
    SELECT
      *
    FROM
      pre_suc_nodes
    UNION ALL
    SELECT
      *
    FROM
      padding_last_row
  ),
  pre_suc_count AS (
    SELECT
      preceding_node,
      succeeding_node,
      COUNT(succeeding_node) OVER (
        PARTITION BY
          preceding_node,
          succeeding_node
      ) AS succeeding_node_count,
      COUNT(preceding_node) OVER (
        PARTITION BY
          preceding_node
      ) AS preceding_node_count
    FROM
      fully_pre_suc_node
  ),
  pre_suc_prob AS (
    SELECT
      preceding_node,
      succeeding_node,
      succeeding_node_count / (preceding_node_count + CAST('{alpha}' AS INT64)) AS transition_prob
    FROM
      pre_suc_count
  )
SELECT DISTINCT
  --  needs source, target, weight columns
  preceding_node AS source,
  succeeding_node AS target,
  transition_prob AS weight
FROM
  pre_suc_prob
ORDER BY
  preceding_node,
  transition_prob DESC
