import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object yelpApp {
  def main(args: Array[String]): Unit = {

    val currentDirectory = new java.io.File(".").getCanonicalPath

    println(currentDirectory)
    val spark = SparkSession.builder
      .appName("yelpApp")
      .master("local[4]").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val reviewSchema = new StructType()
      .add("review_id", StringType)
      .add("user_id", StringType)
      .add("business_id", StringType)
      .add("stars", IntegerType)
      .add("date", TimestampType)
      .add("text", StringType)

    val reviewDF = spark.read
                        .schema(reviewSchema)
                        .json(currentDirectory + "/../dataset/yelp_academic_dataset_review.json")

    val positiveReviewDF = reviewDF.where(col("stars") > 4)

    val groupByUserDF = positiveReviewDF.groupBy("user_id")
      .agg(count("business_id").as("length"), collect_set("business_id").as("business_ids_list"))
    val filteredReviewCount = groupByUserDF.where(col("length") > 5)

    //groupByUserDF.show(truncate=false)
    println("total user count: " + groupByUserDF.count())

    val explodedDF = filteredReviewCount.select(col("user_id"), explode(col("business_ids_list")).as("business_id")).cache()
    //explodedDF.show(truncate=false)
    println(explodedDF.count())

    explodedDF.write.format("jdbc")
      .option("url", "jdbc:mysql://localhost:3306/yelp")
      .option("dbtable", "data")
      .option("user", "root")
      .option("password", "example")
      .option("createTableColumnTypes", "user_id VARCHAR(128), business_id VARCHAR(128)")
      .mode("overwrite")
      .save()

    val usersDF = explodedDF.select("user_id").distinct
    usersDF.show()

    usersDF.write.format("jdbc")
      .option("url", "jdbc:mysql://localhost:3306/yelp")
      .option("dbtable", "users")
      .option("user", "root")
      .option("password", "example")
      .option("createTableColumnTypes", "user_id VARCHAR(128)")
      .mode("overwrite")
      .save()

    val businessesDF = explodedDF.select("business_id").distinct
    businessesDF.show()

    businessesDF.write.format("jdbc")
      .option("url", "jdbc:mysql://localhost:3306/yelp")
      .option("dbtable", "business_ids")
      .option("user", "root")
      .option("password", "example")
      .option("createTableColumnTypes", "business_id VARCHAR(128)")
      .mode("overwrite")
      .save()
  }
}
