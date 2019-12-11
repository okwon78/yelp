import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object yelpApp {
  def main(args: Array[String]): Unit = {

    val currentDirectory = new java.io.File(".").getCanonicalPath

    println(currentDirectory)
    val spark = SparkSession.builder
      .appName("apmalPopularPrdApp")
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
      .agg(count("business_id").as("length"), collect_list("business_id").as("business_ids"))

    val trainDF = groupByUserDF.where(col("length") > 5).cache()
    trainDF.show(truncate=false)
    println("total user count: " + trainDF.count())

    val explodedDF = trainDF.select(col("user_id"), explode(col("business_ids")).as("business_id")).cache()
    explodedDF.show(truncate=false)
    println(explodedDF.count())

    val groupByBusiness = explodedDF.groupBy("business_id").agg(count("user_id").as("length"), collect_list("user_id"))
    val filteredReviewCount = groupByBusiness.where(col("length") > 3)
    filteredReviewCount.show(truncate=false)
    println("total business_id count: " + filteredReviewCount.count())
  }
}
