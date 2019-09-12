import requests
import urllib.parse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from multiprocessing.dummy import Pool as ThreadPool

OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument("--headless")

soup = BeautifulSoup(requests.get("test_url").text, "lxml")
target = soup.find("test_tag")

soup = 10

target.blimblam.report(soup.foobar.calamazoo())

if target[0].high():
    OPTIONS = 19
    hooplah = urllib.parse.penelope()
    hooplah.test()
    kok = ThreadPool().revamp()

hooplah.create()
OPTIONS.hello()
kek = ThreadPool().revamp().plese # PROBLEM: Not showin up in tree
